# AI Buckingham — pressure_gradient

Run: `20261124T120120Z_1faeb540`  
Status: completed  
Mode: live

- Interprets U_tau/U_inf as a balance between viscous scale (nu) and outer-layer momentum/geometry: larger MKE and theta raise normalized friction, while larger delta or delta* reduce it — an empirical fit (GPR R2≈0.91).
- RAG grounding: exact algebraic identity with the friction Reynolds number: U_tau/U_inf = Re_tau / Re_delta (Re_tau = u_tau delta/nu, Re_delta = U_inf delta/nu). Re_tau compares viscous length to the outer scale.

Full structured evidence is in state.json, events.jsonl and evidence/.
