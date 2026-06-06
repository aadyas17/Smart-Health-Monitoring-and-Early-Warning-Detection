# Water-Borne Disease Surveillance System - Use Case Diagrams

## 1) UML Arrow Rules Used

Use these exact relation rules in standard UML use case diagrams:

- Actor association:
  - Solid line
  - Example: Citizen --> (Submit Complaint)
- Include:
  - Dashed arrow from base use case to mandatory reused use case
  - Label: <<include>>
  - Example: (Submit Complaint) ..> (Validate Complaint Data) : <<include>>
- Extend:
  - Dashed arrow from optional/conditional extension use case to base use case
  - Label: <<extend>>
  - Example: (Trigger High Risk Alert) ..> (Generate Risk Prediction) : <<extend>>

Rule summary:
- Use <<include>> for always-required behavior.
- Use <<extend>> for conditional behavior (only in some situations).

---

## 2) Main Use Case Diagram (All 3 Actors)

```mermaid
flowchart TB
  citizen[Citizen]
  asha[ASHA Worker]
  official[Health Official]

  subgraph system[Water-Borne Disease Surveillance System]
    direction TB
    uc1((Register / Login))
    uc2((Submit Complaint))
    uc3((Track Complaint Status))
    uc4((Submit Field Data))
    uc5((Generate Risk Prediction))
    uc6((Send Medium/High Alert))
    uc7((View Dashboard & Update Cases))
  end

  citizen --- uc1
  citizen --- uc2
  citizen --- uc3

  asha --- uc1
  asha --- uc4

  official --- uc1
  official --- uc7

  uc2 -. "<<include>>" .-> uc1
  uc3 -. "<<include>>" .-> uc1
  uc4 -. "<<include>>" .-> uc5
  uc6 -. "<<extend>>" .-> uc5
  uc7 -. "<<include>>" .-> uc1
```

---

## 3) Risk Prediction Use Case (Include/Extend Focus)

This is the exact behavior implemented in your prediction endpoint:
- Input validation always happens.
- Model prediction always happens.
- Prediction record is always stored.
- Medium and High email notifications happen only conditionally.
- Low risk does not trigger email.

```mermaid
flowchart TB
  asha[ASHA Worker]
  official[Health Official]
  citizen[Citizen]

  subgraph pred[Prediction and Alert Subsystem]
    direction TB
    p1((Submit Field Data))
    p2((Validate Parameters))
    p3((Run SVM Prediction))
    p4((Store Prediction))
    p5((Return Risk Level))
    p6((Send Medium/High Email))
  end

  asha --- p1
  asha --- p5
  official --- p6
  citizen --- p5

  p1 -. "<<include>>" .-> p2
  p1 -. "<<include>>" .-> p3
  p1 -. "<<include>>" .-> p4
  p1 -. "<<include>>" .-> p5
  p6 -. "<<extend>>" .-> p3
```

---

## 4) Mapping to Your Current Project Endpoints

Citizen:
- Register Account -> /api/local/register
- Login -> /api/local/login
- Submit Complaint -> /api/complaint/submit
- View Own Complaints, Track Status -> /api/complaints/<local_id>

ASHA Worker:
- Register ASHA Account -> /api/asha/register
- Login ASHA -> /api/asha/login
- Submit Field Data and Request Prediction -> /api/predict
- View Prediction History -> /api/asha/predictions
- View Area Complaints -> /api/asha/complaints

Health Official:
- Admin Login -> /api/admin/login
- View Dashboard Statistics -> /api/admin/stats
- View All Predictions -> /api/admin/predictions
- View All Complaints -> /api/admin/complaints
- View Registered Citizens -> /api/admin/locals
- View Registered ASHA Workers -> /api/admin/asha
- View Login Sessions -> /api/admin/sessions
- Update Complaint Status -> /api/admin/update_complaint_status/<id>
- Delete System Records -> /api/admin/delete/<type>/<id>
