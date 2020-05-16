## DDPG Algorithm - Reacher Continuous Control
### Learning Algorithm
The Learning algorithm used in this project is a modification of the implementation used in the [pendulum environment](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).

DPG is a policy based learning algorithm in which the agent will learn from the nonprocessed observation spaces without knowing the environment. In contrast to a DQN which learns directly through a gradient method which estimates the weights of the optimal policy. 

DDPG employs an Actor-Critic model in which the Critic model learns the state-value function and uses this to determine how the Actor's policy model should change. The Actor learns from the continuous space without the needs for many data samples since it can rely on the critic to give it feedback on good and bad actions.

However, stability could be a problem with this approach. You can end up with a model that learns well and crashes after a few episodes. To mitigate these, there are several techniques that can be employed like, Gradient Clipping, Soft Target Update, Twin local/ target networks, and Replay Buffer. Reply Buffer is most critical as it allows the DDPG agent to learn offline by gathering experiences collected from the environment agents and sample experiences from a large memory buffer across experiences.

### Model Architecture
The Udacity provided DDPG code in PyTorch was used and adapted for this 20 agent environment.

The algorithm uses two deep neural networks (actor-critic) with the following struture:
- Actor    
    - Hidden: (input, 256)  - ReLU
    - BatchNormalization: (256)
    - Hidden: (256, 256)    - ReLU
    - Output: (256, 4)      - TanH   # action_size=4

- Critic
    - Hidden: (input, 256)              - ReLU
    - BatchNormalization: (256)
    - Hidden: (256 + action_size, 256)  - ReLU
    - Output: (256, 1)                  - Linear


### Hyperparameters
    - BUFFER_SIZE = int(1e6)  # replay buffer size
    - BATCH_SIZE = 128         # minibatch size
    - GAMMA = 0.99            # discount factor
    - TAU = 1e-3              # for soft update of target parameters
    - LR_ACTOR = 1e-4         # learning rate of the actor
    - LR_CRITIC = 1e-4        # learning rate of the critic
    - WEIGHT_DECAY = 0.0      # L2 weight decay

## Results 

The model was able to achieve the goal with 364 episodes.

Episode 1	Score: 0.19	Windowed Average Score: 0.19
Episode 2	Score: 1.48	Windowed Average Score: 0.83
Episode 3	Score: 0.86	Windowed Average Score: 0.84
Episode 4	Score: 0.10	Windowed Average Score: 0.66
Episode 5	Score: 0.05	Windowed Average Score: 0.54
Episode 6	Score: 0.17	Windowed Average Score: 0.47
Episode 7	Score: 0.49	Windowed Average Score: 0.48
Episode 8	Score: 0.80	Windowed Average Score: 0.52
Episode 9	Score: 0.43	Windowed Average Score: 0.51
Episode 10	Score: 0.23	Windowed Average Score: 0.48
Episode 11	Score: 0.34	Windowed Average Score: 0.47
Episode 12	Score: 0.48	Windowed Average Score: 0.47
Episode 13	Score: 0.19	Windowed Average Score: 0.45
Episode 14	Score: 0.41	Windowed Average Score: 0.44
Episode 15	Score: 0.85	Windowed Average Score: 0.47
Episode 16	Score: 0.68	Windowed Average Score: 0.48
Episode 17	Score: 0.56	Windowed Average Score: 0.49
Episode 18	Score: 1.00	Windowed Average Score: 0.52
Episode 19	Score: 2.11	Windowed Average Score: 0.60
Episode 20	Score: 0.63	Windowed Average Score: 0.60
Episode 21	Score: 1.56	Windowed Average Score: 0.65
Episode 22	Score: 1.72	Windowed Average Score: 0.70
Episode 23	Score: 1.91	Windowed Average Score: 0.75
Episode 24	Score: 1.15	Windowed Average Score: 0.77
Episode 25	Score: 2.91	Windowed Average Score: 0.85
Episode 26	Score: 1.91	Windowed Average Score: 0.89
Episode 27	Score: 1.29	Windowed Average Score: 0.91
Episode 28	Score: 1.21	Windowed Average Score: 0.92
Episode 29	Score: 1.12	Windowed Average Score: 0.93
Episode 30	Score: 0.54	Windowed Average Score: 0.91
Episode 31	Score: 2.29	Windowed Average Score: 0.96
Episode 32	Score: 1.58	Windowed Average Score: 0.98
Episode 33	Score: 2.66	Windowed Average Score: 1.03
Episode 34	Score: 1.00	Windowed Average Score: 1.03
Episode 35	Score: 2.38	Windowed Average Score: 1.07
Episode 36	Score: 1.25	Windowed Average Score: 1.07
Episode 37	Score: 1.69	Windowed Average Score: 1.09
Episode 38	Score: 3.47	Windowed Average Score: 1.15
Episode 39	Score: 2.29	Windowed Average Score: 1.18
Episode 40	Score: 2.94	Windowed Average Score: 1.22
Episode 41	Score: 1.71	Windowed Average Score: 1.23
Episode 42	Score: 2.02	Windowed Average Score: 1.25
Episode 43	Score: 3.49	Windowed Average Score: 1.31
Episode 44	Score: 2.27	Windowed Average Score: 1.33
Episode 45	Score: 2.82	Windowed Average Score: 1.36
Episode 46	Score: 1.54	Windowed Average Score: 1.36
Episode 47	Score: 2.38	Windowed Average Score: 1.39
Episode 48	Score: 2.90	Windowed Average Score: 1.42
Episode 49	Score: 4.14	Windowed Average Score: 1.47
Episode 50	Score: 2.93	Windowed Average Score: 1.50
Episode 51	Score: 1.73	Windowed Average Score: 1.51
Episode 52	Score: 2.13	Windowed Average Score: 1.52
Episode 53	Score: 3.50	Windowed Average Score: 1.56
Episode 54	Score: 3.41	Windowed Average Score: 1.59
Episode 55	Score: 3.80	Windowed Average Score: 1.63
Episode 56	Score: 2.05	Windowed Average Score: 1.64
Episode 57	Score: 2.22	Windowed Average Score: 1.65
Episode 58	Score: 2.33	Windowed Average Score: 1.66
Episode 59	Score: 1.51	Windowed Average Score: 1.66
Episode 60	Score: 2.59	Windowed Average Score: 1.67
Episode 61	Score: 3.10	Windowed Average Score: 1.70
Episode 62	Score: 1.11	Windowed Average Score: 1.69
Episode 63	Score: 2.35	Windowed Average Score: 1.70
Episode 64	Score: 4.02	Windowed Average Score: 1.73
Episode 65	Score: 3.39	Windowed Average Score: 1.76
Episode 66	Score: 2.08	Windowed Average Score: 1.76
Episode 67	Score: 2.98	Windowed Average Score: 1.78
Episode 68	Score: 3.58	Windowed Average Score: 1.81
Episode 69	Score: 4.88	Windowed Average Score: 1.85
Episode 70	Score: 2.78	Windowed Average Score: 1.87
Episode 71	Score: 4.19	Windowed Average Score: 1.90
Episode 72	Score: 1.13	Windowed Average Score: 1.89
Episode 73	Score: 2.78	Windowed Average Score: 1.90
Episode 74	Score: 2.16	Windowed Average Score: 1.90
Episode 75	Score: 1.84	Windowed Average Score: 1.90
Episode 76	Score: 3.50	Windowed Average Score: 1.92
Episode 77	Score: 3.43	Windowed Average Score: 1.94
Episode 78	Score: 3.85	Windowed Average Score: 1.97
Episode 79	Score: 5.87	Windowed Average Score: 2.02
Episode 80	Score: 4.82	Windowed Average Score: 2.05
Episode 81	Score: 4.26	Windowed Average Score: 2.08
Episode 82	Score: 2.37	Windowed Average Score: 2.08
Episode 83	Score: 3.79	Windowed Average Score: 2.10
Episode 84	Score: 4.12	Windowed Average Score: 2.13
Episode 85	Score: 3.19	Windowed Average Score: 2.14
Episode 86	Score: 3.27	Windowed Average Score: 2.15
Episode 87	Score: 4.66	Windowed Average Score: 2.18
Episode 88	Score: 2.08	Windowed Average Score: 2.18
Episode 89	Score: 3.24	Windowed Average Score: 2.19
Episode 90	Score: 4.43	Windowed Average Score: 2.22
Episode 91	Score: 5.34	Windowed Average Score: 2.25
Episode 92	Score: 5.13	Windowed Average Score: 2.28
Episode 93	Score: 3.22	Windowed Average Score: 2.29
Episode 94	Score: 2.29	Windowed Average Score: 2.29
Episode 95	Score: 5.62	Windowed Average Score: 2.33
Episode 96	Score: 2.61	Windowed Average Score: 2.33
Episode 97	Score: 5.93	Windowed Average Score: 2.37
Episode 98	Score: 2.15	Windowed Average Score: 2.37
Episode 99	Score: 3.30	Windowed Average Score: 2.38
Episode 100	Score: 4.60	Windowed Average Score: 2.40
Episode 101	Score: 2.63	Windowed Average Score: 2.42
Episode 102	Score: 3.40	Windowed Average Score: 2.44
Episode 103	Score: 4.02	Windowed Average Score: 2.47
Episode 104	Score: 4.50	Windowed Average Score: 2.52
Episode 105	Score: 7.19	Windowed Average Score: 2.59
Episode 106	Score: 3.14	Windowed Average Score: 2.62
Episode 107	Score: 4.71	Windowed Average Score: 2.66
Episode 108	Score: 4.10	Windowed Average Score: 2.69
Episode 109	Score: 5.65	Windowed Average Score: 2.75
Episode 110	Score: 4.44	Windowed Average Score: 2.79
Episode 111	Score: 5.95	Windowed Average Score: 2.84
Episode 112	Score: 7.86	Windowed Average Score: 2.92
Episode 113	Score: 5.92	Windowed Average Score: 2.98
Episode 114	Score: 3.58	Windowed Average Score: 3.01
Episode 115	Score: 7.63	Windowed Average Score: 3.07
Episode 116	Score: 4.68	Windowed Average Score: 3.11
Episode 117	Score: 5.00	Windowed Average Score: 3.16
Episode 118	Score: 5.11	Windowed Average Score: 3.20
Episode 119	Score: 7.78	Windowed Average Score: 3.26
Episode 120	Score: 4.72	Windowed Average Score: 3.30
Episode 121	Score: 3.98	Windowed Average Score: 3.32
Episode 122	Score: 2.33	Windowed Average Score: 3.33
Episode 123	Score: 5.08	Windowed Average Score: 3.36
Episode 124	Score: 9.15	Windowed Average Score: 3.44
Episode 125	Score: 3.62	Windowed Average Score: 3.45
Episode 126	Score: 8.42	Windowed Average Score: 3.51
Episode 127	Score: 4.78	Windowed Average Score: 3.55
Episode 128	Score: 9.33	Windowed Average Score: 3.63
Episode 129	Score: 5.53	Windowed Average Score: 3.67
Episode 130	Score: 8.13	Windowed Average Score: 3.75
Episode 131	Score: 7.62	Windowed Average Score: 3.80
Episode 132	Score: 10.80	Windowed Average Score: 3.89
Episode 133	Score: 3.69	Windowed Average Score: 3.90
Episode 134	Score: 6.31	Windowed Average Score: 3.96
Episode 135	Score: 6.35	Windowed Average Score: 4.00
Episode 136	Score: 9.06	Windowed Average Score: 4.07
Episode 137	Score: 10.69	Windowed Average Score: 4.16
Episode 138	Score: 13.31	Windowed Average Score: 4.26
Episode 139	Score: 10.37	Windowed Average Score: 4.34
Episode 140	Score: 6.65	Windowed Average Score: 4.38
Episode 141	Score: 6.84	Windowed Average Score: 4.43
Episode 142	Score: 10.52	Windowed Average Score: 4.52
Episode 143	Score: 6.84	Windowed Average Score: 4.55
Episode 144	Score: 5.61	Windowed Average Score: 4.58
Episode 145	Score: 15.39	Windowed Average Score: 4.71
Episode 146	Score: 7.01	Windowed Average Score: 4.76
Episode 147	Score: 7.78	Windowed Average Score: 4.82
Episode 148	Score: 7.05	Windowed Average Score: 4.86
Episode 149	Score: 5.21	Windowed Average Score: 4.87
Episode 150	Score: 12.26	Windowed Average Score: 4.96
Episode 151	Score: 10.66	Windowed Average Score: 5.05
Episode 152	Score: 5.90	Windowed Average Score: 5.09
Episode 153	Score: 12.62	Windowed Average Score: 5.18
Episode 154	Score: 10.28	Windowed Average Score: 5.25
Episode 155	Score: 9.25	Windowed Average Score: 5.31
Episode 156	Score: 4.32	Windowed Average Score: 5.33
Episode 157	Score: 11.99	Windowed Average Score: 5.43
Episode 158	Score: 6.86	Windowed Average Score: 5.47
Episode 159	Score: 8.46	Windowed Average Score: 5.54
Episode 160	Score: 17.97	Windowed Average Score: 5.69
Episode 161	Score: 6.32	Windowed Average Score: 5.73
Episode 162	Score: 11.72	Windowed Average Score: 5.83
Episode 163	Score: 6.60	Windowed Average Score: 5.88
Episode 164	Score: 8.61	Windowed Average Score: 5.92
Episode 165	Score: 11.72	Windowed Average Score: 6.00
Episode 166	Score: 13.22	Windowed Average Score: 6.12
Episode 167	Score: 12.15	Windowed Average Score: 6.21
Episode 168	Score: 6.73	Windowed Average Score: 6.24
Episode 169	Score: 3.85	Windowed Average Score: 6.23
Episode 170	Score: 19.16	Windowed Average Score: 6.39
Episode 171	Score: 10.03	Windowed Average Score: 6.45
Episode 172	Score: 15.70	Windowed Average Score: 6.60
Episode 173	Score: 13.06	Windowed Average Score: 6.70
Episode 174	Score: 16.51	Windowed Average Score: 6.84
Episode 175	Score: 9.73	Windowed Average Score: 6.92
Episode 176	Score: 11.79	Windowed Average Score: 7.00
Episode 177	Score: 25.65	Windowed Average Score: 7.23
Episode 178	Score: 9.35	Windowed Average Score: 7.28
Episode 179	Score: 14.65	Windowed Average Score: 7.37
Episode 180	Score: 14.54	Windowed Average Score: 7.47
Episode 181	Score: 8.90	Windowed Average Score: 7.51
Episode 182	Score: 16.89	Windowed Average Score: 7.66
Episode 183	Score: 12.03	Windowed Average Score: 7.74
Episode 184	Score: 10.33	Windowed Average Score: 7.80
Episode 185	Score: 22.66	Windowed Average Score: 8.00
Episode 186	Score: 22.48	Windowed Average Score: 8.19
Episode 187	Score: 12.38	Windowed Average Score: 8.27
Episode 188	Score: 12.51	Windowed Average Score: 8.37
Episode 189	Score: 9.14	Windowed Average Score: 8.43
Episode 190	Score: 11.69	Windowed Average Score: 8.50
Episode 191	Score: 15.30	Windowed Average Score: 8.60
Episode 192	Score: 10.13	Windowed Average Score: 8.65
Episode 193	Score: 16.69	Windowed Average Score: 8.79
Episode 194	Score: 17.37	Windowed Average Score: 8.94
Episode 195	Score: 20.74	Windowed Average Score: 9.09
Episode 196	Score: 10.01	Windowed Average Score: 9.16
Episode 197	Score: 14.54	Windowed Average Score: 9.25
Episode 198	Score: 9.39	Windowed Average Score: 9.32
Episode 199	Score: 13.27	Windowed Average Score: 9.42
Episode 200	Score: 13.30	Windowed Average Score: 9.51
Episode 201	Score: 17.74	Windowed Average Score: 9.66
Episode 202	Score: 20.08	Windowed Average Score: 9.83
Episode 203	Score: 15.34	Windowed Average Score: 9.94
Episode 204	Score: 15.55	Windowed Average Score: 10.05
Episode 205	Score: 11.49	Windowed Average Score: 10.09
Episode 206	Score: 19.80	Windowed Average Score: 10.26
Episode 207	Score: 21.20	Windowed Average Score: 10.42
Episode 208	Score: 13.37	Windowed Average Score: 10.52
Episode 209	Score: 19.24	Windowed Average Score: 10.65
Episode 210	Score: 8.28	Windowed Average Score: 10.69
Episode 211	Score: 20.22	Windowed Average Score: 10.83
Episode 212	Score: 14.56	Windowed Average Score: 10.90
Episode 213	Score: 19.29	Windowed Average Score: 11.04
Episode 214	Score: 21.28	Windowed Average Score: 11.21
Episode 215	Score: 11.64	Windowed Average Score: 11.25
Episode 216	Score: 10.52	Windowed Average Score: 11.31
Episode 217	Score: 22.47	Windowed Average Score: 11.49
Episode 218	Score: 19.46	Windowed Average Score: 11.63
Episode 219	Score: 23.91	Windowed Average Score: 11.79
Episode 220	Score: 21.59	Windowed Average Score: 11.96
Episode 221	Score: 18.78	Windowed Average Score: 12.11
Episode 222	Score: 21.44	Windowed Average Score: 12.30
Episode 223	Score: 20.88	Windowed Average Score: 12.46
Episode 224	Score: 18.21	Windowed Average Score: 12.55
Episode 225	Score: 23.24	Windowed Average Score: 12.74
Episode 226	Score: 21.24	Windowed Average Score: 12.87
Episode 227	Score: 17.99	Windowed Average Score: 13.00
Episode 228	Score: 18.22	Windowed Average Score: 13.09
Episode 229	Score: 18.33	Windowed Average Score: 13.22
Episode 230	Score: 21.16	Windowed Average Score: 13.35
Episode 231	Score: 21.21	Windowed Average Score: 13.49
Episode 232	Score: 20.36	Windowed Average Score: 13.58
Episode 233	Score: 19.86	Windowed Average Score: 13.74
Episode 234	Score: 24.21	Windowed Average Score: 13.92
Episode 235	Score: 21.29	Windowed Average Score: 14.07
Episode 236	Score: 23.33	Windowed Average Score: 14.21
Episode 237	Score: 27.04	Windowed Average Score: 14.38
Episode 238	Score: 18.70	Windowed Average Score: 14.43
Episode 239	Score: 19.28	Windowed Average Score: 14.52
Episode 240	Score: 24.31	Windowed Average Score: 14.70
Episode 241	Score: 17.66	Windowed Average Score: 14.81
Episode 242	Score: 23.95	Windowed Average Score: 14.94
Episode 243	Score: 10.80	Windowed Average Score: 14.98
Episode 244	Score: 21.18	Windowed Average Score: 15.14
Episode 245	Score: 18.40	Windowed Average Score: 15.17
Episode 246	Score: 16.28	Windowed Average Score: 15.26
Episode 247	Score: 26.23	Windowed Average Score: 15.44
Episode 248	Score: 23.78	Windowed Average Score: 15.61
Episode 249	Score: 22.28	Windowed Average Score: 15.78
Episode 250	Score: 17.18	Windowed Average Score: 15.83
Episode 251	Score: 28.40	Windowed Average Score: 16.01
Episode 252	Score: 28.38	Windowed Average Score: 16.23
Episode 253	Score: 27.35	Windowed Average Score: 16.38
Episode 254	Score: 16.29	Windowed Average Score: 16.44
Episode 255	Score: 19.00	Windowed Average Score: 16.54
Episode 256	Score: 34.93	Windowed Average Score: 16.84
Episode 257	Score: 24.38	Windowed Average Score: 16.97
Episode 258	Score: 30.73	Windowed Average Score: 17.21
Episode 259	Score: 18.24	Windowed Average Score: 17.30
Episode 260	Score: 14.89	Windowed Average Score: 17.27
Episode 261	Score: 16.57	Windowed Average Score: 17.38
Episode 262	Score: 18.92	Windowed Average Score: 17.45
Episode 263	Score: 14.18	Windowed Average Score: 17.52
Episode 264	Score: 24.87	Windowed Average Score: 17.69
Episode 265	Score: 24.61	Windowed Average Score: 17.81
Episode 266	Score: 29.65	Windowed Average Score: 17.98
Episode 267	Score: 29.08	Windowed Average Score: 18.15
Episode 268	Score: 23.10	Windowed Average Score: 18.31
Episode 269	Score: 31.05	Windowed Average Score: 18.58
Episode 270	Score: 19.08	Windowed Average Score: 18.58
Episode 271	Score: 25.65	Windowed Average Score: 18.74
Episode 272	Score: 29.66	Windowed Average Score: 18.88
Episode 273	Score: 31.77	Windowed Average Score: 19.07
Episode 274	Score: 29.33	Windowed Average Score: 19.19
Episode 275	Score: 31.22	Windowed Average Score: 19.41
Episode 276	Score: 21.12	Windowed Average Score: 19.50
Episode 277	Score: 34.40	Windowed Average Score: 19.59
Episode 278	Score: 25.23	Windowed Average Score: 19.75
Episode 279	Score: 30.01	Windowed Average Score: 19.90
Episode 280	Score: 20.89	Windowed Average Score: 19.97
Episode 281	Score: 25.75	Windowed Average Score: 20.13
Episode 282	Score: 19.44	Windowed Average Score: 20.16
Episode 283	Score: 31.39	Windowed Average Score: 20.35
Episode 284	Score: 31.20	Windowed Average Score: 20.56
Episode 285	Score: 37.12	Windowed Average Score: 20.71
Episode 286	Score: 19.89	Windowed Average Score: 20.68
Episode 287	Score: 24.09	Windowed Average Score: 20.80
Episode 288	Score: 23.09	Windowed Average Score: 20.90
Episode 289	Score: 23.71	Windowed Average Score: 21.05
Episode 290	Score: 17.55	Windowed Average Score: 21.11
Episode 291	Score: 33.13	Windowed Average Score: 21.29
Episode 292	Score: 26.32	Windowed Average Score: 21.45
Episode 293	Score: 34.51	Windowed Average Score: 21.63
Episode 294	Score: 36.83	Windowed Average Score: 21.82
Episode 295	Score: 29.35	Windowed Average Score: 21.91
Episode 296	Score: 19.20	Windowed Average Score: 22.00
Episode 297	Score: 29.99	Windowed Average Score: 22.15
Episode 298	Score: 30.19	Windowed Average Score: 22.36
Episode 299	Score: 34.16	Windowed Average Score: 22.57
Episode 300	Score: 34.69	Windowed Average Score: 22.78
Episode 301	Score: 27.90	Windowed Average Score: 22.89
Episode 302	Score: 25.64	Windowed Average Score: 22.94
Episode 303	Score: 31.50	Windowed Average Score: 23.10
Episode 304	Score: 23.94	Windowed Average Score: 23.19
Episode 305	Score: 36.05	Windowed Average Score: 23.43
Episode 306	Score: 23.86	Windowed Average Score: 23.47
Episode 307	Score: 31.32	Windowed Average Score: 23.57
Episode 308	Score: 28.92	Windowed Average Score: 23.73
Episode 309	Score: 33.35	Windowed Average Score: 23.87
Episode 310	Score: 36.64	Windowed Average Score: 24.15
Episode 311	Score: 22.15	Windowed Average Score: 24.17
Episode 312	Score: 28.17	Windowed Average Score: 24.31
Episode 313	Score: 30.25	Windowed Average Score: 24.42
Episode 314	Score: 30.44	Windowed Average Score: 24.51
Episode 315	Score: 27.65	Windowed Average Score: 24.67
Episode 316	Score: 22.08	Windowed Average Score: 24.79
Episode 317	Score: 17.60	Windowed Average Score: 24.74
Episode 318	Score: 31.05	Windowed Average Score: 24.85
Episode 319	Score: 26.98	Windowed Average Score: 24.88
Episode 320	Score: 34.99	Windowed Average Score: 25.02
Episode 321	Score: 28.38	Windowed Average Score: 25.11
Episode 322	Score: 31.88	Windowed Average Score: 25.22
Episode 323	Score: 32.73	Windowed Average Score: 25.34
Episode 324	Score: 36.04	Windowed Average Score: 25.52
Episode 325	Score: 25.05	Windowed Average Score: 25.53
Episode 326	Score: 36.26	Windowed Average Score: 25.68
Episode 327	Score: 26.16	Windowed Average Score: 25.77
Episode 328	Score: 37.53	Windowed Average Score: 25.96
Episode 329	Score: 36.23	Windowed Average Score: 26.14
Episode 330	Score: 32.31	Windowed Average Score: 26.25
Episode 331	Score: 30.03	Windowed Average Score: 26.34
Episode 332	Score: 37.36	Windowed Average Score: 26.51
Episode 333	Score: 37.52	Windowed Average Score: 26.68
Episode 334	Score: 29.97	Windowed Average Score: 26.74
Episode 335	Score: 37.20	Windowed Average Score: 26.90
Episode 336	Score: 32.48	Windowed Average Score: 26.99
Episode 337	Score: 32.81	Windowed Average Score: 27.05
Episode 338	Score: 24.48	Windowed Average Score: 27.11
Episode 339	Score: 30.41	Windowed Average Score: 27.22
Episode 340	Score: 35.54	Windowed Average Score: 27.33
Episode 341	Score: 36.55	Windowed Average Score: 27.52
Episode 342	Score: 35.72	Windowed Average Score: 27.64
Episode 343	Score: 34.08	Windowed Average Score: 27.87
Episode 344	Score: 35.59	Windowed Average Score: 28.02
Episode 345	Score: 36.89	Windowed Average Score: 28.20
Episode 346	Score: 25.00	Windowed Average Score: 28.29
Episode 347	Score: 31.38	Windowed Average Score: 28.34
Episode 348	Score: 35.32	Windowed Average Score: 28.45
Episode 349	Score: 34.38	Windowed Average Score: 28.58
Episode 350	Score: 26.69	Windowed Average Score: 28.67
Episode 351	Score: 29.01	Windowed Average Score: 28.68
Episode 352	Score: 35.39	Windowed Average Score: 28.75
Episode 353	Score: 32.92	Windowed Average Score: 28.80
Episode 354	Score: 35.79	Windowed Average Score: 29.00
Episode 355	Score: 22.60	Windowed Average Score: 29.03
Episode 356	Score: 31.54	Windowed Average Score: 29.00
Episode 357	Score: 27.72	Windowed Average Score: 29.03
Episode 358	Score: 32.64	Windowed Average Score: 29.05
Episode 359	Score: 37.15	Windowed Average Score: 29.24
Episode 360	Score: 32.69	Windowed Average Score: 29.42
Episode 361	Score: 35.46	Windowed Average Score: 29.61
Episode 362	Score: 28.12	Windowed Average Score: 29.70
Episode 363	Score: 36.44	Windowed Average Score: 29.92
Episode 364	Score: 34.29	Windowed Average Score: 30.02


Environment solved in 364 episodes!	tWindowed Average Score: 30.02

![Training](https://github.com/nitink12/DeepReinforcementLearningNanoDegree/blob/master/P2_Continuous-Control/images/training.png)


## future improvements
1. Continue training for a few more iterations
2. Improve the DDPG alogorithm by applying Priority Experience Replay.
3. Solving the Version 2 of the environment with 20 identical agents
