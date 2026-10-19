from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predict_ml import generate_predictions

app = FastAPI(title="Modelo ML - Objetivos de venta")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/objetivos/generar")
def generar_objetivos():
    """Entrena el modelo con el histórico actual de la BD y devuelve la
    predicción del próximo mes. No escribe nada en la base de datos:
    la persistencia queda a cargo de la API principal (Node.js)."""
    try:
        result, next_period, skipped = generate_predictions()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = {
        "period": next_period.strftime("%Y-%m-%d"),
        "objetivos": result.assign(
            period=result["period"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }
    if skipped is not None and len(skipped) > 0:
        payload["omitidos"] = skipped.to_dict(orient="records")

    return JSONResponse(payload)
