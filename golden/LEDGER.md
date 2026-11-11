# Golden set ledger

Updated 2026-11-10 16:43. Score = auto (0-40) + judge (0-60).

| iteration | tasks done | mean score | notes |
|---|---|---|---|
| iter_01 | 6/6 | None | VOID: network/DNS outage (ENOTFOUND) killed every agent mid-build at ~01:25; scores not meaningful |
| iter_02 | 6/6 | 88.7 | fixes from iter_01 partials: networkx, visual merge, scope rule; harness outage-resume |
| iter_03 | 6/6 | 89.8 | playbook v3: full-demo evidence (DEMO_SECONDS), showcase videos, settling at home pose, camera unblocking, salience hint |
| iter_04 | 6/6 | 91.0 | playbook v4: world.py dynamics hook, fixed thresholds, REPORT.md, simplest-design defaults, env.py reaches goal, workspa |
| iter_05 | 6/6 | 90.9 | playbook v5 + golden v2 (user states success criteria): box-fit scaling, world.py, task-object occlusion check, FPV fix, |

## iter_01

| task | total | auto | judge | shots | smoke | min | $ | top problem |
|---|---|---|---|---|---|---|---|---|
| arm_kitchen | 0 | 0.0 | None | 3 | None | 31.9 | 5.89 |  |
| drone_gates | 0 | 0.0 | None | 3 | None | 33.9 | 7.33 |  |
| isaac_synthetic | 0 | 0.0 | None | 3 | None | 24.1 | 0.0 |  |
| quadruped_trail | 18.0 | 18.0 | None | 3 | FAIL | 33.8 | 6.85 |  |
| sd_intersection | 0 | 0.0 | None | 3 | None | 32.2 | 4.93 |  |
| warehouse_amr | 0 | 0.0 | None | 3 | None | 23.7 | 0.0 |  |

## iter_02

| task | total | auto | judge | shots | smoke | min | $ | top problem |
|---|---|---|---|---|---|---|---|---|
| arm_kitchen | 87.0 | 37.0 | 50.0 | 2 | PASS_WITH_WARNINGS | 18.6 | 5.27 | At the home pose the overhead camera is partly blocked by the arm. The final report says the arm was folded ba |
| drone_gates | 87.5 | 37.0 | 50.5 | 2 | PASS_WITH_WARNINGS | 20.8 | 6.06 | The regrade filmstrip only covers t=0–4 s of a ~25 s flight, so it shows the takeoff but never shows the drone |
| isaac_synthetic | 86.0 | 37.0 | 49.0 | 2 | PASS_WITH_WARNINGS | 68.3 | 15.96 | The camera-rig body (a black box on a short post) is visible in the user's requested down-aisle camera and in  |
| quadruped_trail | 89.0 | 40.0 | 49.0 | 2 | PASS | 18.8 | 6.02 | Rocks barely register in wide shots. Most of the 10 rocks are 7-14 cm stepping stones that render as grey spec |
| sd_intersection | 88.5 | 37.0 | 51.5 | 2 | PASS_WITH_WARNINGS | 33.5 | 9.59 | The regrade filmstrip's tracking camera is blocked by a tree crown in most frames, and it only covers 4 s. You |
| warehouse_amr | 94.0 | 40.0 | 54.0 | 2 | PASS | 21.2 | 4.19 | The generic smoke rollout and filmstrip cover only the first 4 s with a fixed camera. They show the AMR leavin |

## iter_03

| task | total | auto | judge | shots | smoke | min | $ | top problem |
|---|---|---|---|---|---|---|---|---|
| factory_sorting | 93.0 | 40.0 | 53.0 | 2 | PASS | 35.7 | 6.33 | The shot-1 brief committed to a friction-driven moving belt surface and didn't ask about the belt mechanism. T |
| harbor_boat | 86.5 | 37.0 | 49.5 | 2 | PASS_WITH_WARNINGS | 110.6 | None | Buoyancy and drag live in a Python hydro.apply() that every caller has to invoke before each step. The scene i |
| humanoid_gym | 90.5 | 40.0 | 50.5 | 2 | PASS | 25.7 | 6.59 | The user named 'joint limits respected' as a task-level check, but no task check tests it. Nothing verifies th |
| lab_bench | 89.5 | 40.0 | 49.5 | 2 | PASS | 74.9 | None | There is no final report. The shot-2 reply is only a rate-limit message, so the user never got a summary of wh |
| mars_rover | 88.5 | 37.0 | 51.5 | 2 | PASS_WITH_WARNINGS | 33.8 | 6.44 | The env.py baseline 'heading-to-lander' policy never reaches the lander (it ends 25 m away and gets stuck on a |
| robot_soccer | 90.5 | 40.0 | 50.5 | 2 | PASS | 25.0 | 5.3 | The user specifically asked to check that the ball rolls with realistic friction, but no task check measures t |

## iter_04

| task | total | auto | judge | shots | smoke | min | $ | top problem |
|---|---|---|---|---|---|---|---|---|
| arm_kitchen | 91.5 | 40.0 | 51.5 | 2 | PASS | 20.4 | 6.15 | The arm's home pose puts the upper arm right across the overhead camera's view. It covers about a quarter of t |
| cartpole_minimal | 91.0 | 38.0 | 53.0 | 2 | PASS | 19.2 | 2.13 | The filmstrip's 'workspace' camera is fixed while the cart moves, so the cart drifts out of frame and is partl |
| cloth_table | 83.4 | 36.38 | 47.0 | 2 | PASS_WITH_WARNINGS | 93.9 | 14.43 | Shot 2 hit the 90-minute time limit ($14.43). The agent never wrote a final report: shot2_reply.md is a single |
| drone_gates | 93.5 | 40.0 | 53.5 | 2 | PASS | 24.3 | 6.06 | The FPV camera image, which is also the RL image observation, has its bottom ~20% filled by a flat white slab  |
| home_fetch | 92.5 | 40.0 | 52.5 | 2 | PASS | 31.8 | 8.02 | Uniform (no-stretch) scaling left several objects off the planned size: coffee table 1.43 x 0.96 m (planned 1. |
| orchard_drone | 94.0 | 40.0 | 54.0 | 2 | PASS | 25.0 | 4.24 | Brief defaulted to 6 rows and a 3-row back-and-forth demo, which was more than the request needed. The user ha |

## iter_05

| task | total | auto | judge | shots | smoke | min | $ | top problem |
|---|---|---|---|---|---|---|---|---|
| harbor_boat | 90.0 | 40.0 | 50.0 | 2 | PASS | 51.7 | 10.02 | Boat mass changed from the briefed ~900 kg to 2000 kg, but the final reply's 'Changes from the plan' list leav |
| isaac_synthetic | 91.5 | 40.0 | 51.5 | 2 | PASS | 50.1 | 12.08 | The ceiling is hidden in the MuJoCo renders, so every aisle view has a flat grey/black gradient band across th |
| lab_bench | 91.5 | 40.0 | 51.5 | 2 | PASS | 25.2 | 6.6 | The auto_iso_back camera shows only an empty wall/floor corner, so this view is useless for review. |
| mars_rover | 90.5 | 40.0 | 50.5 | 2 | PASS | 29.6 | 7.38 | The goal-distance metric changed between the brief and the build. The brief promised the rover would end 'with |
| quadruped_trail | 90.5 | 40.0 | 50.5 | 2 | PASS | 21.1 | 5.77 | Stair climbability is shown only by geometry measurements and by placing the robot on the landing. The robot n |
| sd_intersection | 91.5 | 40.0 | 51.5 | 2 | PASS | 26.2 | 6.41 | The tracking/follow camera used in the showcase is blocked in some frames. At about 4.7 s a large green slab ( |
