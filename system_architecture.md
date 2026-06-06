# System Architecture Diagram

```mermaid
flowchart LR
  C[Citizen]
  A[ASHA Worker]
  H[Health Official / Admin]

  subgraph P[Presentation Layer]
    CP[Citizen Portal]
    AP[ASHA Portal]
    AD[Admin Dashboard]
  end

  subgraph B[Backend / Application Layer]
    GW[API Gateway]
    AUTH[Authentication Service]
    REP[Complaint / Report Management]
    VAL[Field Validation Service]
    ALERT[Alert Service]
  end

  subgraph M[ML Prediction Layer]
    PRE[Data Preprocessing]
    MOD[ML Model Module]
    RISK[Risk Prediction]
  end

  subgraph D[Data / Persistence Layer]
    DB[(Central Database)]
    USERS[User Roles]
    RPTS[Complaints & Reports]
    PREDS[Predictions]
    ALTS[Alerts]
  end

  SMTP[Email Service]

  C --> CP
  A --> AP
  H --> AD

  CP --> GW
  AP --> GW
  AD --> GW

  GW --> AUTH
  GW --> REP
  GW --> VAL
  GW --> ALERT

  GW --> PRE
  PRE --> MOD
  MOD --> RISK

  AUTH --> DB
  REP --> DB
  VAL --> DB
  ALERT --> SMTP
  ALERT --> DB
  RISK --> DB

  DB --- USERS
  DB --- RPTS
  DB --- PREDS
  DB --- ALTS
```

This diagram matches your project better:

- `Citizen Portal` maps to the local user pages and complaint flow.
- `ASHA Portal` maps to ASHA login, prediction, and complaint tracking.
- `Admin Dashboard` maps to admin login and admin reports.
- `Complaint / Report Management` covers complaint submission and viewing.
- `ML Prediction Layer` represents the saved model and scaler used in prediction.
- `Email Service` represents the alert emails sent after medium or high risk predictions.

Use this version if you want a clean, easy-to-explain system architecture for your report.
```