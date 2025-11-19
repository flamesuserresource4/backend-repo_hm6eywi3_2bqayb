import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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
    # stringify _id for JSON
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
