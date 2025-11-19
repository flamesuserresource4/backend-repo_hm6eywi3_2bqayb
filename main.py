import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents
from schemas import SCHEMAS_REGISTRY, Machine, ProductionOrder, MaintenanceTicket, Alert, AutomationLog

app = FastAPI(title="Manufacturing AI Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Manufacturing AI Dashboard Backend is running"}

@app.get("/schema")
def get_schema_registry():
    # Return schema names and field definitions for the frontend viewer
    out: Dict[str, Any] = {}
    for name, model in SCHEMAS_REGISTRY.items():
        fields = {k: str(v.annotation) for k, v in model.model_fields.items()}  # type: ignore
        out[name] = {"fields": fields}
    return out

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response

# ---------- Domain Endpoints ----------

@app.post("/machines")
def create_machine(payload: Machine):
    inserted_id = create_document("machine", payload)
    return {"id": inserted_id}

@app.get("/machines")
def list_machines():
    items = get_documents("machine")
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items

@app.post("/orders")
def create_order(payload: ProductionOrder):
    inserted_id = create_document("productionorder", payload)
    return {"id": inserted_id}

@app.get("/orders")
def list_orders():
    items = get_documents("productionorder")
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items

@app.post("/maintenance")
def create_ticket(payload: MaintenanceTicket):
    inserted_id = create_document("maintenanceticket", payload)
    return {"id": inserted_id}

@app.get("/maintenance")
def list_tickets():
    items = get_documents("maintenanceticket")
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items

@app.post("/alerts")
def create_alert(payload: Alert):
    inserted_id = create_document("alert", payload)
    return {"id": inserted_id}

@app.get("/alerts")
def list_alerts():
    items = get_documents("alert")
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items

@app.post("/automation/log")
def log_automation(payload: AutomationLog):
    inserted_id = create_document("automationlog", payload)
    return {"id": inserted_id}

@app.get("/automation/logs")
def list_automation_logs():
    items = get_documents("automationlog")
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items

# ---------- Advanced Analytics & Optimization ----------

class MaintenanceRisk(BaseModel):
    machine_code: str
    risk_score: float = Field(ge=0, le=100)
    reason: str
    recommended_date: Optional[datetime] = None

@app.get("/predict/maintenance")
def predict_maintenance():
    """Heuristic risk scoring using uptime, OEE, ticket history, and last maintenance."""
    machines = list_machines()
    tickets = list_tickets()
    risks: List[MaintenanceRisk] = []

    # Pre-aggregate ticket severities per machine
    sev_weight = {"low": 5, "medium": 12, "high": 22, "critical": 35}
    t_by_machine: Dict[str, float] = {}
    open_tickets = {t.get("machine_code"): t for t in tickets if t.get("status") != "resolved"}
    for t in tickets:
        code = t.get("machine_code")
        t_by_machine[code] = t_by_machine.get(code, 0) + sev_weight.get(t.get("severity", "low"), 5)

    for m in machines:
        code = m.get("code") or m.get("machine_code") or "unknown"
        uptime = float(m.get("uptime_hours", 0) or 0)
        oee = float(m.get("oee", 0) or 0)
        last = m.get("last_maintenance")
        base = 20.0
        base += min(uptime / 100.0, 30.0)  # more uptime -> higher wear
        base += max(0.0, (60 - min(60.0, oee))) * 0.5  # low OEE -> higher risk
        base += t_by_machine.get(code, 0)
        if m.get("status") in ["down", "maintenance"]:
            base += 20
        base = max(0.0, min(100.0, base))
        # Recommend date based on risk bucket
        days = 21
        if base >= 80: days = 1
        elif base >= 60: days = 3
        elif base >= 40: days = 7
        elif base >= 20: days = 14
        rec_date = datetime.now(timezone.utc) + timedelta(days=days)
        reason = f"uptime={uptime}h, oee={oee}%, open_tickets={'yes' if code in open_tickets else 'no'}"
        risks.append(MaintenanceRisk(machine_code=code, risk_score=round(base, 1), reason=reason, recommended_date=rec_date))

    # Sort high risk first
    risks_sorted = sorted([r.model_dump() for r in risks], key=lambda x: x["risk_score"], reverse=True)
    return risks_sorted

class SimulateShiftRequest(BaseModel):
    hours: float = 8.0
    speed_per_machine: Optional[Dict[str, float]] = None  # units/hour per machine code

@app.post("/simulate/shift")
def simulate_shift(req: SimulateShiftRequest):
    """Simple digital twin: estimate completions for in-progress orders over horizon."""
    machines = list_machines()
    orders = list_orders()

    speed = req.speed_per_machine or {}
    # Default base speed if not provided
    for m in machines:
        code = m.get("code")
        if code not in speed:
            # base speed proportional to OEE
            oee = float(m.get("oee", 70) or 70)
            speed[code] = max(5.0, oee / 2.0)  # units/hour

    timeline: List[Dict[str, Any]] = []
    total_completed = 0
    for o in orders:
        if o.get("status") in ("completed", "cancelled"):
            continue
        assigned = o.get("machine_code")
        qty_remaining = max(0, int(o.get("quantity", 0)) - int(o.get("completed", 0)))
        if qty_remaining <= 0:
            continue
        # choose machine: assigned if available, else fastest
        chosen_code = assigned if assigned in speed else max(speed.keys(), key=lambda c: speed[c] if c in speed else 0)
        rate = speed.get(chosen_code, 10.0)
        projected = min(qty_remaining, int(rate * req.hours))
        done_pct = round(100 * projected / max(1, qty_remaining))
        total_completed += projected
        timeline.append({
            "order_no": o.get("order_no"),
            "machine_code": chosen_code,
            "projected_completed": projected,
            "remaining_before": qty_remaining,
            "done_pct": done_pct
        })

    return {
        "horizon_hours": req.hours,
        "total_projected": total_completed,
        "per_order": timeline,
    }

class OptimizeScheduleRequest(BaseModel):
    objective: Optional[str] = Field("throughput", description="throughput | due_date | balance")

@app.post("/schedule/optimize")
def optimize_schedule(_: OptimizeScheduleRequest):
    """Greedy scheduler: assign orders to best machine by status and speed proxy (OEE)."""
    machines = list_machines()
    orders = list_orders()

    # Build speed proxy by OEE
    speed = {}
    available = []
    for m in machines:
        if m.get("status") in ("down", "maintenance"):
            continue
        code = m.get("code")
        speed[code] = max(5.0, float(m.get("oee", 60) or 60) / 2.0)
        available.append(code)
    if not available:
        raise HTTPException(status_code=400, detail="No available machines to schedule")

    plan = []
    for o in orders:
        if o.get("status") in ("completed", "cancelled"):
            continue
        # prefer assigned machine if healthy
        assigned = o.get("machine_code")
        if assigned in available:
            chosen = assigned
        else:
            chosen = max(available, key=lambda c: speed.get(c, 0))
        rate = speed.get(chosen, 10.0)
        qty_remaining = max(0, int(o.get("quantity", 0)) - int(o.get("completed", 0)))
        est_hours = round(qty_remaining / max(1.0, rate), 2)
        plan.append({
            "order_no": o.get("order_no"),
            "machine_code": chosen,
            "est_hours": est_hours
        })

    # sort by shortest processing time (SPT)
    plan.sort(key=lambda x: x["est_hours"]) 
    return {"assignments": plan}

# ---------- AI/Groq proxy ----------

class GroqChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = "llama-3.1-70b-versatile"
    temperature: Optional[float] = 0.2

@app.post("/ai/insights")
def ai_insights(req: GroqChatRequest):
    """
    Calls Groq Chat Completions API and returns a concise insight for manufacturing KPIs.
    Requires GROQ_API_KEY env var to be set.
    """
    import requests
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": req.model,
        "messages": [
            {"role": "system", "content": "You are a manufacturing operations analyst. Be concise and practical."},
            {"role": "user", "content": req.prompt},
        ],
        "temperature": req.temperature,
        "max_tokens": 256,
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq API error: {r.text[:200]}")
    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except Exception:
        content = str(data)[:2000]
    return {"insight": content}

@app.post("/ai/root_cause")
def ai_root_cause():
    """Summarize likely root causes from recent alerts, tickets, and machine stats using Groq."""
    import requests
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured")

    machines = list_machines()[:10]
    alerts = list_alerts()[:10]
    tickets = list_tickets()[:10]

    context = {
        "machines": [{"code": m.get("code"), "status": m.get("status"), "oee": m.get("oee"), "uptime": m.get("uptime_hours")} for m in machines],
        "alerts": [{"level": a.get("level"), "message": a.get("message")} for a in alerts],
        "tickets": [{"machine": t.get("machine_code"), "severity": t.get("severity"), "status": t.get("status")} for t in tickets]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = (
        "You are a senior reliability engineer. Based on the JSON context, suggest the 3 most likely root causes "
        "impacting throughput and the exact next diagnostic checks to perform. Provide bullet points with machine codes where relevant.\n\n"
        f"Context: {context}"
    )

    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "system", "content": "Be precise, use shop-floor language, and limit to 120 words."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq API error: {r.text[:200]}")
    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except Exception:
        content = str(data)[:2000]
    return {"root_cause": content}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
