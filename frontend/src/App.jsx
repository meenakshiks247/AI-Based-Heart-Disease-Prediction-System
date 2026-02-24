import React, { useState } from 'react'

/* ─── model definitions ─── */
const MODELS = {
    clinical: {
        label: 'Clinical (Cleveland) Model',
        endpoint: 'http://127.0.0.1:8000/api/predict/',
        note: null,
        defaults: {
            age: 54, sex: 1, cp: 0, trestbps: 130, chol: 250, fbs: 0,
            restecg: 0, thalach: 150, exang: 0, oldpeak: 1.0, slope: 1, ca: 0, thal: 2,
        },
        labels: {
            age: 'Age', sex: 'Sex (1 = Male, 0 = Female)', cp: 'Chest Pain Type (0–3)',
            trestbps: 'Resting Blood Pressure', chol: 'Cholesterol (mg/dl)',
            fbs: 'Fasting Blood Sugar > 120 (1/0)', restecg: 'Resting ECG (0–2)',
            thalach: 'Max Heart Rate', exang: 'Exercise‑Induced Angina (1/0)',
            oldpeak: 'ST Depression (Oldpeak)', slope: 'Slope of Peak ST (0–2)',
            ca: 'Major Vessels Colored (0–4)', thal: 'Thalassemia (0–3)',
        },
        floatFields: ['oldpeak'],
    },
    cardio: {
        label: 'Population (Cardio) Model',
        endpoint: 'http://127.0.0.1:8000/api/predict/cardio/',
        note: 'Population model trained on 68k samples — uses a different feature set.',
        defaults: {
            age: 50, sex: 2, height: 168, weight: 62, systolic_bp: 120,
            diastolic_bp: 80, cholesterol: 1, gluc: 1, smoke: 0, alco: 0, active: 1,
        },
        labels: {
            age: 'Age (years)', sex: 'Sex (1 = Female, 2 = Male)',
            height: 'Height (cm)', weight: 'Weight (kg)',
            systolic_bp: 'Systolic BP (ap_hi)', diastolic_bp: 'Diastolic BP (ap_lo)',
            cholesterol: 'Cholesterol (1–3)', gluc: 'Glucose (1–3)',
            smoke: 'Smoking (0/1)', alco: 'Alcohol (0/1)', active: 'Active (0/1)',
        },
        floatFields: ['age', 'weight'],
    },
}

export default function App() {
    const [modelKey, setModelKey] = useState('clinical')
    const model = MODELS[modelKey]

    const [formData, setFormData] = useState({ ...model.defaults })
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [fieldErrors, setFieldErrors] = useState({})

    /* switch model — reset form, result, errors */
    const handleModelChange = (e) => {
        const key = e.target.value
        setModelKey(key)
        setFormData({ ...MODELS[key].defaults })
        setResult(null)
        setError(null)
        setFieldErrors({})
    }

    /* safely parse a numeric value */
    const safeParse = (name, raw) => {
        if (raw === '' || raw === '-') return ''
        const parsed = model.floatFields.includes(name) ? parseFloat(raw) : parseInt(raw, 10)
        return Number.isNaN(parsed) ? '' : parsed
    }

    /* update a single field */
    const handleChange = (e) => {
        const { name, value } = e.target
        const parsed = safeParse(name, value)
        setFormData((prev) => ({ ...prev, [name]: parsed }))
        /* clear the field error as soon as the user types a valid value */
        if (parsed !== '' && !Number.isNaN(parsed)) {
            setFieldErrors((prev) => { const copy = { ...prev }; delete copy[name]; return copy })
        }
    }

    /* validate all fields before submit */
    const validate = () => {
        const errors = {}
        for (const key of Object.keys(model.defaults)) {
            if (formData[key] === '' || formData[key] === null || formData[key] === undefined || Number.isNaN(formData[key])) {
                errors[key] = `${model.labels[key]} is required`
            }
        }
        setFieldErrors(errors)
        return Object.keys(errors).length === 0
    }

    /* call the backend */
    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!validate()) return          /* ← block submit if any field is bad */
        setLoading(true)
        setResult(null)
        setError(null)

        try {
            const response = await fetch(model.endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            })

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`)
            }

            const data = await response.json()
            setResult(data)
        } catch (err) {
            setError('Prediction failed. Please ensure the backend is running.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="app">
            <header className="header">
                <h1>❤️ Heart Disease Predictor</h1>
                <p className="subtitle">
                    Enter patient data below and click <strong>Predict</strong> to assess
                    heart disease risk.
                </p>
            </header>

            <form className="form" onSubmit={handleSubmit}>
                {/* ─── model selector ─── */}
                <div className="field model-selector">
                    <label htmlFor="model-select">Model</label>
                    <select
                        id="model-select"
                        value={modelKey}
                        onChange={handleModelChange}
                    >
                        {Object.entries(MODELS).map(([key, m]) => (
                            <option key={key} value={key}>{m.label}</option>
                        ))}
                    </select>
                </div>

                {model.note && <p className="model-note">{model.note}</p>}

                <div className="grid">
                    {Object.keys(model.defaults).map((key) => (
                        <div className={`field ${fieldErrors[key] ? 'field--error' : ''}`} key={key}>
                            <label htmlFor={key}>{model.labels[key]}</label>
                            <input
                                id={key}
                                name={key}
                                type="number"
                                step={model.floatFields.includes(key) ? '0.1' : '1'}
                                value={formData[key]}
                                onChange={handleChange}
                                required
                            />
                            {fieldErrors[key] && (
                                <span className="field-error">{fieldErrors[key]}</span>
                            )}
                        </div>
                    ))}
                </div>

                <button className="btn" type="submit" disabled={loading}>
                    {loading ? 'Predicting…' : '🔍 Predict'}
                </button>
            </form>

            {/* ─── result card ─── */}
            {result && (
                <div
                    className={`result-card ${result.prediction === 1 ? 'high-risk' : 'low-risk'
                        }`}
                >
                    <h2>
                        {result.prediction === 1
                            ? 'High Risk ❤️'
                            : 'Low Risk ✅'}
                    </h2>
                    <p className="probability">
                        Probability:{' '}
                        <strong>{(result.probability * 100).toFixed(1)}%</strong>
                    </p>
                    <p className="model">Model: {result.model_name}</p>
                </div>
            )}

            {/* ─── error banner ─── */}
            {error && <div className="error-banner">{error}</div>}

            <style>{`
        /* ─── reset & base ─── */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
          background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
          min-height: 100vh;
          color: #e0e0e0;
        }

        .app {
          max-width: 740px;
          margin: 0 auto;
          padding: 40px 24px 60px;
        }

        /* ─── header ─── */
        .header {
          text-align: center;
          margin-bottom: 36px;
        }

        .header h1 {
          font-size: 2rem;
          background: linear-gradient(90deg, #f7797d, #fbd786, #c6ffdd);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .subtitle {
          margin-top: 8px;
          color: #b0aec1;
          font-size: 0.95rem;
        }

        /* ─── form grid ─── */
        .form {
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          padding: 32px;
          backdrop-filter: blur(12px);
        }

        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 20px;
        }

        .field {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .field label {
          font-size: 0.82rem;
          font-weight: 600;
          color: #c5c3d4;
          text-transform: uppercase;
          letter-spacing: 0.4px;
        }

        .field input {
          padding: 10px 12px;
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.08);
          color: #f0f0f0;
          font-size: 1rem;
          transition: border-color 0.2s, box-shadow 0.2s;
          outline: none;
        }

        .field input:focus {
          border-color: #7f5af0;
          box-shadow: 0 0 0 3px rgba(127, 90, 240, 0.25);
        }

        /* ─── model selector ─── */
        .model-selector {
          margin-bottom: 20px;
        }

        .model-selector select {
          width: 100%;
          padding: 10px 12px;
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.08);
          color: #f0f0f0;
          font-size: 1rem;
          outline: none;
          cursor: pointer;
          transition: border-color 0.2s, box-shadow 0.2s;
          appearance: none;
          -webkit-appearance: none;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%23999' fill='none' stroke-width='1.5'/%3E%3C/svg%3E");
          background-repeat: no-repeat;
          background-position: right 12px center;
        }

        .model-selector select:focus {
          border-color: #7f5af0;
          box-shadow: 0 0 0 3px rgba(127, 90, 240, 0.25);
        }

        .model-selector select option {
          background: #24243e;
          color: #f0f0f0;
        }

        .model-note {
          margin-bottom: 18px;
          padding: 10px 14px;
          border-radius: 8px;
          background: rgba(127, 90, 240, 0.1);
          border: 1px solid rgba(127, 90, 240, 0.25);
          color: #b0aec1;
          font-size: 0.85rem;
          text-align: center;
        }

        /* ─── inline validation errors ─── */
        .field--error input {
          border-color: #ff453a;
          box-shadow: 0 0 0 2px rgba(255, 69, 58, 0.25);
        }

        .field-error {
          font-size: 0.75rem;
          color: #ff6b6b;
          margin-top: 2px;
        }

        /* ─── button ─── */
        .btn {
          display: block;
          width: 100%;
          margin-top: 28px;
          padding: 14px;
          font-size: 1.05rem;
          font-weight: 700;
          color: #fff;
          background: linear-gradient(135deg, #7f5af0, #2cb67d);
          border: none;
          border-radius: 10px;
          cursor: pointer;
          transition: opacity 0.2s, transform 0.15s;
        }

        .btn:hover:not(:disabled) {
          opacity: 0.9;
          transform: translateY(-1px);
        }

        .btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }

        /* ─── result card ─── */
        .result-card {
          margin-top: 32px;
          padding: 28px;
          border-radius: 16px;
          text-align: center;
          animation: slideUp 0.35s ease-out;
        }

        .result-card.high-risk {
          background: rgba(255, 69, 58, 0.15);
          border: 1px solid rgba(255, 69, 58, 0.4);
        }

        .result-card.low-risk {
          background: rgba(48, 209, 88, 0.15);
          border: 1px solid rgba(48, 209, 88, 0.4);
        }

        .result-card h2 {
          font-size: 1.6rem;
          margin-bottom: 12px;
        }

        .probability {
          font-size: 1.1rem;
          margin-bottom: 4px;
        }

        .model {
          font-size: 0.85rem;
          color: #999;
        }

        /* ─── error ─── */
        .error-banner {
          margin-top: 24px;
          padding: 14px 20px;
          border-radius: 10px;
          background: rgba(255, 69, 58, 0.12);
          border: 1px solid rgba(255, 69, 58, 0.35);
          color: #ff6b6b;
          text-align: center;
          font-weight: 500;
        }

        /* ─── animation ─── */
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        /* ─── responsive ─── */
        @media (max-width: 480px) {
          .grid { grid-template-columns: 1fr; }
          .header h1 { font-size: 1.5rem; }
          .form { padding: 20px; }
        }
      `}</style>
        </div>
    )
}
