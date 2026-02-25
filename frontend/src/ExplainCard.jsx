import React, { useState } from 'react'

/* ─── small inline SVG icons ─── */
const ArrowUp = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="12" y1="19" x2="12" y2="5" /><polyline points="5 12 12 5 19 12" />
  </svg>
)

const ArrowDown = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="12" y1="5" x2="12" y2="19" /><polyline points="19 12 12 19 5 12" />
  </svg>
)

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)

const ChartIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
)

const CloseIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
)

/* ─── Friendly feature labels ─── */
const FEATURE_LABELS = {
  age: 'Age', sex: 'Sex', cp: 'Chest Pain Type', trestbps: 'Resting BP',
  chol: 'Cholesterol', fbs: 'Fasting Blood Sugar', restecg: 'Resting ECG',
  thalach: 'Max Heart Rate', exang: 'Exercise Angina', oldpeak: 'ST Depression',
  slope: 'ST Slope', ca: 'Vessels Colored', thal: 'Thalassemia',
  height: 'Height', weight: 'Weight', systolic_bp: 'Systolic BP',
  diastolic_bp: 'Diastolic BP', cholesterol: 'Cholesterol Level',
  gluc: 'Glucose Level', smoke: 'Smoking', alco: 'Alcohol', active: 'Physical Activity',
}

const SHAP_IMAGE_MAP = {
  clinical: 'http://127.0.0.1:8000/static/models/shap_clinical_summary.png',
  cardio: 'http://127.0.0.1:8000/static/models/shap_cardio_summary.png',
}

/* ─── Spinner ─── */
function Spinner() {
  return (
    <div className="explain-spinner" role="status" aria-label="Loading explanation">
      <div className="explain-spinner__ring" />
      <span className="explain-spinner__text">Analyzing risk factors…</span>
    </div>
  )
}

/* ─── Modal for SHAP summary image ─── */
function ImageModal({ src, alt, onClose }) {
  return (
    <div className="explain-modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-label="Global feature importance">
      <div className="explain-modal" onClick={(e) => e.stopPropagation()}>
        <button className="explain-modal__close" onClick={onClose} aria-label="Close modal">
          <CloseIcon />
        </button>
        <img src={src} alt={alt} className="explain-modal__img" />
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   ExplainCard — top SHAP features + recommendations
   ════════════════════════════════════════════════════════════════════ */
export default function ExplainCard({ data, dataset }) {
  const [showModal, setShowModal] = useState(false)

  if (!data) return null

  const { top_features = [], recommendations = [] } = data
  const shapImgSrc = SHAP_IMAGE_MAP[dataset]

  /* Normalise shap magnitudes for bar widths (max = 100%) */
  const maxShap = Math.max(...top_features.map((f) => Math.abs(f.shap)), 0.001)

  return (
    <>
      <div className="explain-card" role="region" aria-label="Explanation and recommendations">

        {/* ── Top Contributing Factors ─── */}
        <section className="explain-section">
          <h3 className="explain-section__title">
            <ChartIcon /> Top Contributing Factors
          </h3>

          <ul className="explain-features" role="list">
            {top_features.map((f, i) => {
              const isIncrease = f.effect === 'increase'
              const pct = Math.round((Math.abs(f.shap) / maxShap) * 100)
              const label = FEATURE_LABELS[f.feature] || f.feature
              return (
                <li key={i} className="explain-feature" role="listitem">
                  <div className="explain-feature__header">
                    <span
                      className={`explain-pill ${isIncrease ? 'explain-pill--risk' : 'explain-pill--safe'}`}
                      title={isIncrease ? 'Increases risk' : 'Decreases risk'}
                    >
                      {isIncrease ? <ArrowUp /> : <ArrowDown />}
                      <span className="explain-pill__label">
                        {isIncrease ? 'Risk ↑' : 'Protective'}
                      </span>
                    </span>
                    <span className="explain-feature__name">{label}</span>
                    <span className="explain-feature__val">
                      {Math.abs(f.shap).toFixed(3)}
                    </span>
                  </div>
                  <div className="explain-bar-track" aria-hidden="true">
                    <div
                      className={`explain-bar-fill ${isIncrease ? 'explain-bar-fill--risk' : 'explain-bar-fill--safe'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </li>
              )
            })}
          </ul>

          {/* View global importance thumbnail */}
          {shapImgSrc && (
            <button
              className="explain-thumb-btn"
              onClick={() => setShowModal(true)}
              title="View global SHAP feature importance plot"
            >
              <ChartIcon />
              <span>View global feature importance</span>
            </button>
          )}
        </section>

        {/* ── Recommendations ─── */}
        {recommendations.length > 0 && (
          <section className="explain-section">
            <h3 className="explain-section__title">
              <CheckIcon /> Actionable Recommendations
            </h3>
            <ul className="explain-recs" role="list">
              {recommendations.map((rec, i) => (
                <li key={i} className="explain-rec" role="listitem">
                  <span className="explain-rec__icon" aria-hidden="true"><CheckIcon /></span>
                  <span className="explain-rec__text">{rec}</span>
                </li>
              ))}
            </ul>
            <p className="explain-disclaimer">
              These suggestions are not medical advice. Please consult a qualified healthcare professional.
            </p>
          </section>
        )}
      </div>

      {/* ── Image modal ─── */}
      {showModal && (
        <ImageModal
          src={shapImgSrc}
          alt={`SHAP summary plot for ${dataset} model — global feature importance`}
          onClose={() => setShowModal(false)}
        />
      )}

      <style>{`
        /* ═══════ ExplainCard styles ═══════ */

        .explain-card {
          margin-top: 20px;
          padding: 28px;
          border-radius: 16px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(12px);
          animation: slideUp 0.35s ease-out;
        }

        .explain-section + .explain-section {
          margin-top: 28px;
          padding-top: 24px;
          border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .explain-section__title {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 1rem;
          font-weight: 700;
          color: #e0e0e0;
          margin-bottom: 18px;
        }

        /* ── feature list ── */
        .explain-features {
          list-style: none;
          display: flex;
          flex-direction: column;
          gap: 14px;
        }

        .explain-feature__header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 5px;
        }

        .explain-feature__name {
          flex: 1;
          font-size: 0.88rem;
          font-weight: 600;
          color: #d0cfe0;
        }

        .explain-feature__val {
          font-size: 0.78rem;
          font-family: 'Consolas', 'Fira Code', monospace;
          color: #999;
          min-width: 48px;
          text-align: right;
        }

        /* coloured pill */
        .explain-pill {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 2px 8px;
          border-radius: 20px;
          font-size: 0.7rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.4px;
          white-space: nowrap;
        }

        .explain-pill--risk {
          background: rgba(255, 69, 58, 0.18);
          color: #ff6b6b;
          border: 1px solid rgba(255, 69, 58, 0.3);
        }

        .explain-pill--safe {
          background: rgba(48, 209, 88, 0.15);
          color: #30d158;
          border: 1px solid rgba(48, 209, 88, 0.3);
        }

        .explain-pill__label { line-height: 1; }

        /* bar visualisation */
        .explain-bar-track {
          height: 6px;
          border-radius: 3px;
          background: rgba(255, 255, 255, 0.06);
          overflow: hidden;
        }

        .explain-bar-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 0.6s ease-out;
        }

        .explain-bar-fill--risk {
          background: linear-gradient(90deg, #ff453a, #ff6b6b);
        }

        .explain-bar-fill--safe {
          background: linear-gradient(90deg, #30d158, #64e886);
        }

        /* thumbnail button */
        .explain-thumb-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          margin-top: 16px;
          padding: 8px 14px;
          border-radius: 8px;
          border: 1px solid rgba(127, 90, 240, 0.3);
          background: rgba(127, 90, 240, 0.1);
          color: #b0aec1;
          font-size: 0.82rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s, color 0.2s;
          font-family: inherit;
        }

        .explain-thumb-btn:hover {
          background: rgba(127, 90, 240, 0.2);
          color: #e0e0e0;
        }

        /* ── recommendations ── */
        .explain-recs {
          list-style: none;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .explain-rec {
          display: flex;
          align-items: flex-start;
          gap: 10px;
          padding: 12px 14px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.07);
        }

        .explain-rec__icon {
          flex-shrink: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          width: 26px;
          height: 26px;
          border-radius: 50%;
          background: rgba(127, 90, 240, 0.15);
          color: #7f5af0;
          margin-top: 1px;
        }

        .explain-rec__text {
          font-size: 0.88rem;
          line-height: 1.55;
          color: #c5c3d4;
        }

        .explain-disclaimer {
          margin-top: 14px;
          font-size: 0.72rem;
          color: #777;
          font-style: italic;
        }

        /* ── spinner ── */
        .explain-spinner {
          margin-top: 20px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
          padding: 24px;
          animation: slideUp 0.3s ease-out;
        }

        .explain-spinner__ring {
          width: 32px;
          height: 32px;
          border: 3px solid rgba(127, 90, 240, 0.2);
          border-top-color: #7f5af0;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        .explain-spinner__text {
          font-size: 0.85rem;
          color: #999;
        }

        /* ── modal ── */
        .explain-modal-backdrop {
          position: fixed;
          inset: 0;
          z-index: 9999;
          background: rgba(0, 0, 0, 0.75);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
          animation: fadeIn 0.2s ease-out;
        }

        .explain-modal {
          position: relative;
          max-width: 900px;
          width: 100%;
          background: #1e1b38;
          border-radius: 16px;
          border: 1px solid rgba(127, 90, 240, 0.3);
          padding: 16px;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }

        .explain-modal__close {
          position: absolute;
          top: 10px;
          right: 10px;
          width: 34px;
          height: 34px;
          display: flex;
          align-items: center;
          justify-content: center;
          border: none;
          background: rgba(255, 255, 255, 0.08);
          color: #b0aec1;
          border-radius: 50%;
          cursor: pointer;
          transition: background 0.2s;
        }

        .explain-modal__close:hover { background: rgba(255, 255, 255, 0.15); }

        .explain-modal__img {
          width: 100%;
          border-radius: 8px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  )
}

export { Spinner as ExplainSpinner }
