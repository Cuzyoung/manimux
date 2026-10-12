# Scheduling and action clocks

The model predicts a chunk. An inference strategy decides when to request another
chunk and how to hand it off. `ActionTimeline` manages the accepted trajectory;
the executor samples references at the command rate. Scheduling does not change
camera acquisition timestamps or pretend an old observation is new.

## Three different counts

- `policy.horizon_policy_steps`: the model's predicted action count.
- `inference.chunk_policy_steps`: a method-dependent execution/query window.
- Model sampler steps: iterations used to generate the prediction, configured in
  the model recipe/framework.

Twelve points spaced at `1/30 s` span an execution window of about 0.4 s.
Sampling the timeline at 100 Hz does not turn that into 0.12 s. Elapsed command
time also does not prove the robot has physically reached a target.

## Serial and asynchronous execution

```yaml
inference:
  algorithm: manimux
  inference_schedule: serial
  action_start_mode: first_step_when_ready
  chunk_policy_steps: 12
  blend_policy_steps: 0
```

Serial means observe → infer → execute the selected prefix → observe again.
Inference still runs in a worker; the control loop remains responsive and holds
its last command while waiting. The accepted prefix starts at commit time, keeps
its original first row and gives the final row a full action interval. Pause
invalidates the old timeline and in-flight response; resuming uses a new observation.
Serial does not wait for measured arrival, and it does not support the asynchronous
`refill_threshold_s` setting. Set `action_start_mode: first_step_when_ready`.
Both inline and process decoding are supported. Process decoding keeps the
request in flight until decoding finishes, then starts the untrimmed prefix at
commit time. Dense IK rows are capped to the selected source prefix before their
timestamps are shifted; inference and decode latency do not discard early rows.

The asynchronous path can request while the old chunk is executing. Delayed
responses are aligned and trimmed by the selected strategy/timeline contract.
RTC additionally requires real sampler conditioning support from the backend;
changing a YAML algorithm name cannot add missing model-side hooks.

## Meaning of `chunk_policy_steps`

| Strategy | Meaning |
| --- | --- |
| `manimux` | Maximum source prefix; asynchronous delay trimming can reduce usable rows |
| `rtc` | Minimum execution window for requesting again, adjusted using predicted delay; old actions continue during inference |
| `paint` | Source-action query trigger; the old chunk continues while the next request runs |
| `act_temporal_ensemble` | Query interval; overlapping predictions can still contribute to the same action time |
| `dvac` | Maximum adaptive execution length |
| AAC / AutoHorizon | No equivalent fixed window; `chunk_policy_steps` is rejected |

The loader maps this field to the corresponding strategy option. Conflicting
public and internal values are rejected. Equal numbers across methods do not
establish equal effective latency, action counts or deployment behavior.

## ACT temporal ensembling

The retained upstream rule orders predictions for the same absolute action time
from oldest to newest and weights them with:

```text
w_i = exp(-coefficient * i) / sum_j exp(-coefficient * j)
```

The upstream default coefficient is `0.01`. ManiMux exposes the query interval:

```yaml
inference:
  algorithm: act_temporal_ensemble
  blend_policy_steps: 0
  temporal_ensemble:
    coefficient: 0.01
    query_interval_policy_steps: 1
```

The interval is in model action steps, not control ticks. Setting it above one
changes the original every-step query cadence and must be reported as ManiMux's
asynchronous adaptation. Timeline seam blending must be zero for this strategy;
executor filtering/limits remain a separate deployment choice.

Source: [official ACT implementation at the audited revision](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L191-L259).

## Specialized methods

### Train-time RTC

`algorithm: train_time_rtc` uses a prefix-trained sampler instead of inference-time
PiGDM/soft-inpaint guidance. `TrainTimeRtcInferenceStrategy` reuses RTC scheduling
and accepted-plan feedback. The pack-plate preset selects
`train_time_rtc.delay_estimator: fixed` and `fixed_prefix_steps: 5`. Every conditioned
request fixes exactly five action rows, regardless of model/decode latency or
warmup samples; the initial-delay and E2E-window settings do not change this count.
First requests, reset and unavailable committed prefixes still use unconditioned
sampling (`prefix_steps: 0`). Submission and acceptance events record
`prefix_delay_estimator: fixed` and `configured_prefix_steps: 5`, alongside the
actual `prefix_steps` on submissions and measured latency on acceptance. A late
response can have some or all fixed rows trimmed at commit; fixing the mask does
not guarantee latency coverage. The configured count must satisfy `2*prefix <= H`
for the RTC scheduler's feasibility constraint.

For dynamic operation, `train_time_rtc.delay_estimator: decode_max` (the generic default)
uses the ordinary ManiMux process-decode forecast (`decode_forecast_size: 10`,
`decode_forecast_mode: max`, bounded below by `expected_decode_s`). Only decode
latencies enter this maximum. At submission, the prefix forecast is:

`ceil(max(0, current_observation_age + last_model_trip + decode_forecast + handoff_margin - first_action_offset) / action_dt)`.

Here `last_model_trip` runs from the previous accepted request's submission to
worker completion, including preparation, queueing, transport and inference, but
excluding IK. This is a causal estimate: ordinary waypoint handoff knows the
current model latency after receiving its response, whereas trained-prefix
sampling needs its prefix before inference starts. One long model request affects
the next prediction; it is not retained for ten predictions. The first-action
offset (or explicit `adapter.execution_offset_s`) aligns the count to source row
0, as in the old waypoint join-row analysis. Neither the prefix length nor the
predicted switch time is a guarantee that inference will finish on time.

`initial_delay_policy_steps` defaults to 5 until a model-trip sample is available;
it is not a clamp or a permanent lower bound. Reset clears model-trip history.
An aggregate warmup E2E duration cannot seed this split estimator. The explicit
`delay_estimator: end_to_end_max` retains the previous behavior, including E2E
warmup calibration and `train_time_rtc.delay_buffer_size` (default 10); that window
setting has no effect in `decode_max` mode. `chunk_policy_steps` keeps RTC's
minimum-query-window meaning and the existing feasibility check.

Submission and acceptance events record `prefix_delay_estimator`, the observation
age, model/decode forecasts, margin, action offset and `predicted_prefix_steps`.
Acceptance also records `request_to_model_ms` and `measured_prefix_steps` from the
same source-row origin. The existing `measured_delay` / `rtc_delay_ms` remain E2E
diagnostics; they do not feed the `decode_max` prefix predictor.

The UMI trained-prefix history strategy reads the committed execution reference
(including dense IK samples) at each new model target timestamp. It never
interpolates the binary mask or extrapolates missing commands. An incomplete
prefix falls back to unconditioned sampling with a recorded reason. First chunks
and reset also have no prefix. Training support such as `[0, 4, 5]` is metadata,
not a clamp on dynamic inference lengths; submission events record
`prefix_in_training_support` when the checkpoint provides this metadata.

The initial integration requires `blend_policy_steps: 0`, `handoff_skip_steps: 0`
and `action_start_mode: drop_infer_latency`. The timeline trims elapsed rows once;
there is no additional crop by prefix length. Waypoint handoff remains an explicit
execution choice. See the [pack-plate deployment example](../deployment/umi-dp-tianji-taccap.md#pack-plate-train-time-rtc).

See [reproduction records](reproductions/README.md) for upstream versions, exact
sampler hooks, adaptations and evidence limits for AAC, PAINT, AutoHorizon and DVAC.
For the client/sampler boundary, read the [XPolicyLab deployment guide](../deployment/xpolicylab.md).
When comparing tuned deployments, record their actual configuration; isolating an
algorithmic effect requires controlling the other relevant choices.
