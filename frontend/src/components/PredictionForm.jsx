import React, { useState, forwardRef } from 'react'

/* ─── SVG icons ─── */
const StethoscopeIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4.8 2.3A2 2 0 0 0 3 4.5v3a6 6 0 0 0 12 0v-3a2 2 0 0 0-1.8-2.2" />
    <path d="M8 15a6 6 0 0 0 6 6h1a4 4 0 0 0 4-4v-3" />
    <circle cx="19" cy="11" r="2" />
    <line x1="5" y1="1" x2="5" y2="4" />
    <line x1="13" y1="1" x2="13" y2="4" />
  </svg>
)

const HeartIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.4 4.6a5.5 5.5 0 0 0-7.8 0L12 5.2l-.6-.6a5.5 5.5 0 0 0-7.8 7.8l.6.6L12 20.8l7.8-7.8.6-.6a5.5 5.5 0 0 0 0-7.8z" />
  </svg>
)

/* ─── model definitions ─── */
export const MODELS = {
  clinical: {
    label: '🩺  Clinical Diagnosis',
    subtitle: 'Detailed assessment using medical test indicators like ECG, blood work and stress tests.',
    icon: StethoscopeIcon,
    tooltip: 'Uses 13 clinical features from the UCI Cleveland dataset (303 patients). Best suited when ECG, blood work, and stress-test results are available.',
    endpoint: 'http://127.0.0.1:8000/api/predict/',
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
    /* split for two-panel layout */
    leftPanel:  { title: 'Health Status', icon: '🫀', fields: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs'] },
    rightPanel: { title: 'Heart Tests',   icon: '📊', fields: ['restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'] },
  },
  cardio: {
    label: '🌍  General Health Screening',
    subtitle: 'Quick screening based on everyday lifestyle and biometric factors.',
    icon: HeartIcon,
    tooltip: 'Uses 11 lifestyle & biometric features from the Kaggle Cardiovascular dataset (68k+ patients). Works with basic info like height, weight, BP, and habits.',
    endpoint: 'http://127.0.0.1:8000/api/predict/cardio/',
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
    leftPanel:  { title: 'Health Status', icon: '🫀', fields: ['age', 'sex', 'height', 'weight', 'systolic_bp', 'diastolic_bp'] },
    rightPanel: { title: 'Lifestyle',     icon: '🏃', fields: ['cholesterol', 'gluc', 'smoke', 'alco', 'active'] },
  },
}

const PredictionForm = forwardRef(function PredictionForm({ onResult, onError }, ref) {
  const [modelKey, setModelKey] = useState('clinical')
  const model = MODELS[modelKey]

  const [formData, setFormData] = useState({ ...model.defaults })
  const [loading, setLoading] = useState(false)
  const [fieldErrors, setFieldErrors] = useState({})

  /* switch model */
  const handleModelChange = (key) => {
    setModelKey(key)
    setFormData({ ...MODELS[key].defaults })
    onResult(null)
    onError(null)
    setFieldErrors({})
  }

  const safeParse = (name, raw) => {
    if (raw === '' || raw === '-') return ''
    const parsed = model.floatFields.includes(name) ? parseFloat(raw) : parseInt(raw, 10)
    return Number.isNaN(parsed) ? '' : parsed
  }

  const handleChange = (e) => {
    const { name, value } = e.target
    const parsed = safeParse(name, value)
    setFormData((prev) => ({ ...prev, [name]: parsed }))
    if (parsed !== '' && !Number.isNaN(parsed)) {
      setFieldErrors((prev) => { const c = { ...prev }; delete c[name]; return c })
    }
  }

  const validate = () => {
    const errs = {}
    for (const key of Object.keys(model.defaults)) {
      if (formData[key] === '' || formData[key] === null || formData[key] === undefined || Number.isNaN(formData[key])) {
        errs[key] = `${model.labels[key]} is required`
      }
    }
    setFieldErrors(errs)
    return Object.keys(errs).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!validate()) return
    setLoading(true)
    onResult(null)
    onError(null)
    try {
      const res = await fetch(model.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      })
      if (!res.ok) throw new Error(`Server responded with status ${res.status}`)
      const data = await res.json()
      onResult(data)
      /* auto-scroll to result */
      setTimeout(() => {
        document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 120)
    } catch {
      onError('Prediction failed. Please ensure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  /* render a group of fields inside a panel card */
  const renderPanel = (panel) => (
    <div className="pf-panel" key={panel.title}>
      <h3 className="pf-panel__heading">
        <span className="pf-panel__icon">{panel.icon}</span> {panel.title}
      </h3>
      <div className="pf-panel__grid">
        {panel.fields.map((key) => (
          <div className={`pf-field ${fieldErrors[key] ? 'pf-field--error' : ''}`} key={key}>
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
            {fieldErrors[key] && <span className="pf-field__err">{fieldErrors[key]}</span>}
          </div>
        ))}
      </div>
    </div>
  )

  return (
    <>
      <section className="pf-section" id="predict" ref={ref}>
        <h2 className="pf-title">Heart Disease Risk Prediction</h2>
        <p className="pf-subtitle">Choose an analysis mode, fill in your details, and get instant AI-powered insights.</p>

        {/* ─── model selector cards ─── */}
        <div className="pf-selector">
          {Object.entries(MODELS).map(([key, m]) => {
            const active = modelKey === key
            return (
              <button
                key={key}
                type="button"
                className={`pf-model ${active ? 'pf-model--active' : ''}`}
                onClick={() => handleModelChange(key)}
              >
                <span className="pf-model__label">{m.label}</span>
                <span className="pf-model__desc">{m.subtitle}</span>
                {active && <span className="pf-model__check">✓</span>}
              </button>
            )
          })}
        </div>

        {/* ─── two-panel form ─── */}
        <form className="pf-form" onSubmit={handleSubmit}>
          <div className="pf-panels">
            {renderPanel(model.leftPanel)}
            {renderPanel(model.rightPanel)}
          </div>

          <button className="pf-submit" type="submit" disabled={loading}>
            {loading ? (
              <span className="pf-spinner" />
            ) : null}
            {loading ? 'Analysing…' : '🔍  Analyse Risk'}
          </button>
        </form>
      </section>

      <style>{`
        /* ===== PREDICTION FORM ===== */
        .pf-section {
          max-width: 1080px;
          margin: 0 auto;
          padding: 80px 28px 60px;
        }
        .pf-title {
          text-align: center;
          font-size: 2rem;
          font-weight: 800;
          color: #fff;
          margin: 0 0 8px;
        }
        .pf-subtitle {
          text-align: center;
          color: #908fa5;
          font-size: 1rem;
          margin: 0 0 36px;
        }

        /* model selector */
        .pf-selector {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-bottom: 32px;
        }
        .pf-model {
          position: relative;
          display: flex;
          flex-direction: column;
          gap: 6px;
          padding: 22px 22px 18px;
          border-radius: 14px;
          border: 1.5px solid rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.04);
          color: #c5c3d4;
          cursor: pointer;
          text-align: left;
          font-family: inherit;
          transition: border-color 0.25s, background 0.25s, box-shadow 0.25s, transform 0.15s;
          outline: none;
        }
        .pf-model:hover {
          background: rgba(255,255,255,0.07);
          border-color: rgba(127,90,240,0.35);
          transform: translateY(-2px);
        }
        .pf-model--active {
          border-color: #7f5af0;
          background: rgba(127,90,240,0.12);
          box-shadow: 0 0 0 3px rgba(127,90,240,0.2), 0 4px 24px rgba(127,90,240,0.15);
        }
        .pf-model__label {
          font-size: 1.05rem;
          font-weight: 700;
          color: #f0f0f0;
        }
        .pf-model__desc {
          font-size: 0.82rem;
          color: #908fa5;
          line-height: 1.4;
        }
        .pf-model--active .pf-model__desc { color: #b0aec1; }
        .pf-model__check {
          position: absolute;
          top: 12px; right: 14px;
          width: 24px; height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          background: #7f5af0;
          color: #fff;
          font-size: 0.75rem;
          font-weight: 700;
        }

        /* form & panels */
        .pf-form {
          /* no extra background — panels provide it */
        }
        .pf-panels {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
          align-items: start;
        }
        .pf-panel {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          padding: 24px;
          backdrop-filter: blur(10px);
        }
        .pf-panel__heading {
          font-size: 1rem;
          font-weight: 700;
          color: #e0e0e0;
          margin: 0 0 20px;
          padding-bottom: 14px;
          border-bottom: 1px solid rgba(255,255,255,0.07);
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .pf-panel__icon { font-size: 1.2rem; }
        .pf-panel__grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
        }

        /* fields */
        .pf-field {
          display: flex;
          flex-direction: column;
          gap: 6px;
          min-width: 0;
        }
        .pf-field label {
          font-size: 0.78rem;
          font-weight: 600;
          color: #b0aec1;
          text-transform: uppercase;
          letter-spacing: 0.4px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .pf-field input {
          width: 100%;
          padding: 10px 14px;
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 8px;
          background: rgba(255,255,255,0.07);
          color: #f0f0f0;
          font-size: 0.95rem;
          transition: border-color 0.2s, box-shadow 0.2s;
          outline: none;
          font-family: inherit;
        }
        .pf-field input:focus {
          border-color: #7f5af0;
          box-shadow: 0 0 0 3px rgba(127,90,240,0.25);
        }
        .pf-field--error input {
          border-color: #ff453a;
          box-shadow: 0 0 0 2px rgba(255,69,58,0.25);
        }
        .pf-field__err {
          font-size: 0.72rem;
          color: #ff6b6b;
        }

        /* submit */
        .pf-submit {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          width: 100%;
          margin-top: 28px;
          padding: 16px;
          font-size: 1.05rem;
          font-weight: 700;
          color: #fff;
          background: linear-gradient(135deg, #7f5af0, #2cb67d);
          border: none;
          border-radius: 12px;
          cursor: pointer;
          transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
          font-family: inherit;
        }
        .pf-submit:hover:not(:disabled) {
          opacity: 0.92;
          transform: translateY(-2px);
          box-shadow: 0 8px 28px rgba(127,90,240,0.35);
        }
        .pf-submit:disabled { opacity: 0.55; cursor: not-allowed; }

        /* spinner */
        .pf-spinner {
          width: 18px; height: 18px;
          border: 2.5px solid rgba(255,255,255,0.3);
          border-top-color: #fff;
          border-radius: 50%;
          animation: pfSpin 0.6s linear infinite;
        }
        @keyframes pfSpin { to { transform: rotate(360deg); } }

        /* responsive */
        @media (max-width: 768px) {
          .pf-selector { grid-template-columns: 1fr; }
          .pf-panels   { grid-template-columns: 1fr; }
          .pf-panel__grid { grid-template-columns: 1fr 1fr; gap: 20px; }
          .pf-title { font-size: 1.5rem; }
        }
        @media (max-width: 480px) {
          .pf-panel__grid { grid-template-columns: 1fr; gap: 16px; }
          .pf-panel { padding: 20px; }
        }
      `}</style>
    </>
  )
})

export default PredictionForm
