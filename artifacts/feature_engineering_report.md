# PathSnap engineered candidate features

This report documents the exact model inputs in `ml/preprocessing/build_shared_dataset.py::FEATURE_COLUMNS`. The HistGradientBoosting trainer imports that list directly as `FEATURE_COLUMNS`; the saved baseline metrics also record the same names. The ordered list contains 17 numeric columns. `trip_id`, `point_idx`, `candidate_arc_id`, `split`, and `label` are metadata/target columns, not model inputs.

## Coordinate and segment conventions

The feature builder centers each trip at the median track longitude and latitude, then uses a local equirectangular approximation:

$$x=(\lambda-\lambda_0)\,111320\cos(\phi_0),\qquad y=(\phi-\phi_0)\,110574,$$

where longitude $\lambda$ and latitude $\phi$ are in degrees and the resulting local coordinates are approximately metres. For a directed arc with projected start $A=(a_x,a_y)$ and end $B=(b_x,b_y)$, define $v=B-A$, $L=\lVert v\rVert$, and for GPS point $P$, the clipped projection fraction and projected point as

$$t=\operatorname{clip}\!\left(\frac{(P-A)\cdot v}{v\cdot v},0,1\right),\qquad Q=A+tv.$$

When $v\cdot v=0$, the implementation uses $t=0$; when $L=0$, signed offset is set to zero. GPS motion uses the preceding track row. The third `.track` field is numeric and increasing on observed data, but its units are not specified; the code therefore uses neutral `time_delta_units` and `gps_speed_m_per_time_unit` names.

## Model input schema

| # | Exact column | Formula / implementation | Units | Why it helps map matching |
|---:|---|---|---|---|
| 1 | `distance_m` | $\lVert P-Q\rVert$, Euclidean distance from the GPS point to its clipped projection on the candidate segment. | Approximate metres | Measures direct spatial fit to the road. |
| 2 | `signed_lateral_offset_m` | $(v_x(P_y-Q_y)-v_y(P_x-Q_x))/L$ for $L>0$, otherwise 0. Sign is relative to the directed arc orientation. | Approximate metres | Distinguishes which side of a directed road the GPS observation lies on; supplements unsigned distance. |
| 3 | `projection_fraction` | $t=\operatorname{clip}(((P-A)\cdot v)/(v\cdot v),0,1)$, with 0 for a zero-length segment. | Dimensionless, [0, 1] | Locates the observation along the candidate edge, including whether it projects near an endpoint. |
| 4 | `distance_to_arc_start_m` | $tL$. | Approximate metres | Provides distance along the edge from its directed start to the projected GPS point. |
| 5 | `distance_to_arc_end_m` | $(1-t)L$. | Approximate metres | Provides remaining along-edge distance to its directed end; useful for continuity between observations. |
| 6 | `arc_length_m` | $L=\lVert B-A\rVert$. | Approximate metres | Encodes road-segment scale, relevant to candidate geometry and transitions. |
| 7 | `candidate_rank_distance` | Zero-based rank after sorting retained candidates by `(distance_m, candidate_arc_id)` ascending. | Integer rank | Expresses relative spatial preference among the point's candidates. |
| 8 | `candidate_count_at_point` | Number of retained candidate arcs for this GPS point after radius/cap selection and reference insertion during training generation. | Integer count | Reflects local candidate ambiguity and road density. In inference, it counts inference candidates, without reference insertion. |
| 9 | `is_nearest_candidate` | 1 if `candidate_rank_distance == 0`, otherwise 0. | Binary | Explicitly marks the closest road while allowing the classifier to learn when nearest is not best. |
| 10 | `is_within_search_radius` | 1 if `distance_m <= candidate_generation.search_radius_m` (configured as 60 m), otherwise 0. | Binary | Marks candidates satisfying the spatial search threshold; nearest fallback candidates can be outside it. |
| 11 | `gps_step_distance_m` | For point index $i>0$, $\lVert P_i-P_{i-1}\rVert$; at the first point, 0. | Approximate metres | Captures observed movement scale, which informs how plausible candidate progression is. |
| 12 | `gps_speed_m_per_time_unit` | For $i>0$, `gps_step_distance_m / max(time_delta_units, 0)` when the positive time delta is available; otherwise 0. First point is 0. | Approximate metres per source time unit (unit unspecified) | Adds motion-rate context without falsely assuming the source timestamp unit. |
| 13 | `time_delta_units` | At $i>0$, $\max(t_i-t_{i-1},0)$ from the third `.track` field; first point is 0. | Source time units (unspecified) | Indicates observation spacing and whether speed/heading context is reliable. |
| 14 | `gps_bearing_valid` | 1 when the previous-step time delta is positive and the step distance exceeds 0.1 m; otherwise 0 (including the first point). | Binary | Identifies whether the heading comparison is supported by actual movement. |
| 15 | `heading_alignment_cos` | If heading is valid, $\cos(\theta_{gps}-\theta_{arc})$, where each bearing is `atan2(delta_y, delta_x)` in projected coordinates and the arc bearing follows its directed endpoints; otherwise 0. | Dimensionless, [-1, 1] | Measures whether motion aligns with the candidate's direction, distinguishing parallel, opposing, and perpendicular roads. |
| 16 | `heading_alignment_sin` | If heading is valid, $\sin(\theta_{gps}-\theta_{arc})$ using the same bearings; otherwise 0. | Dimensionless, [-1, 1] | Preserves the signed angular difference that cosine alone cannot distinguish. |
| 17 | `is_self_loop` | 1 if $L<0.01$ m, otherwise 0. | Binary | Flags degenerate/nearly zero-length edges where projection and bearing geometry are less informative. |

## Verification

The ordered columns above must equal the current `FEATURE_COLUMNS` list in the generator. The trainer selects `data[FEATURE_COLUMNS]` for both fitting and evaluation. Inference candidate generation uses the same feature list and geometric/motion formulas, but does not open `.route`, create labels, or add the training-only `split`/`label` fields.

Candidate generation queries an STRtree with a radius-sized box, then filters by exact point-to-segment distance, sorts by distance and arc ID, and retains up to the configured maximum of 16. If no arc is within the radius, it adds the nearest graph arc; `is_within_search_radius` remains 0 for that fallback. Training label generation may additionally insert the dynamically aligned reference arc; evaluator inference must not and does not do so.