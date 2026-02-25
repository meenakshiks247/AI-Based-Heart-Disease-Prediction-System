import React from 'react'

/**
 * ResultCard — displays prediction result, probability bar, health factors,
 * and recommendations. Fully self-contained with scoped CSS.
 */
export default function ResultCard({ result }) {
  if (!result) return null

  const riskKey =
    result.risk_level === 'High Risk'     ? 'high' :
    result.risk_level === 'Moderate Risk'  ? 'moderate' : 'low'

  const riskLabel = result.risk_level.replace(' Risk', '').toUpperCase()
  const pct       = (result.probability * 100).toFixed(1)

  const riskColors = {
    high:     { main: '#ff6b6b', bg: 'rgba(255,69,58,0.12)',  border: 'rgba(255,69,58,0.35)' },
    moderate: { main: '#ffb300', bg: 'rgba(255,179,0,0.12)',  border: 'rgba(255,179,0,0.35)' },
    low:      { main: '#2cb67d', bg: 'rgba(44,182,125,0.12)', border: 'rgba(44,182,125,0.35)' },
  }
  const c = riskColors[riskKey]

  return (
    <>
      <section className="rc-section" id="result-section">
        {/* ─── Main risk card ─── */}
        <div className="rc-card" style={{ background: c.bg, borderColor: c.border }}>
          <div className="rc-badge" style={{ background: c.main }}>
            {riskKey === 'high' ? '⚠️' : riskKey === 'moderate' ? '⚡' : '✅'}
          </div>

          <h2 className="rc-risk" style={{ color: c.main }}>
            ❤️&nbsp;&nbsp;Risk Level : {riskLabel}
          </h2>

          {/* probability bar */}
          <div className="rc-bar-wrap">
            <div className="rc-bar-track">
              <div
                className="rc-bar-fill"
                style={{ width: `${pct}%`, background: c.main }}
              />
            </div>
            <span className="rc-bar-label" style={{ color: c.main }}>
              {pct}%
            </span>
          </div>

          <p className="rc-prob-text">
            {riskKey === 'high'
              ? 'Elevated cardiac risk detected. Consult a cardiologist promptly.'
              : riskKey === 'moderate'
              ? 'Borderline risk — lifestyle adjustments and monitoring advised.'
              : 'Low risk profile — maintain your healthy habits.'}
          </p>
        </div>

        {/* ─── Health factors + Recommendations ─── */}
        <div className="rc-details">
          {/* positive factors */}
          <div className="rc-detail-card">
            <h3 className="rc-detail-heading rc-detail-heading--pos">
              🟢&nbsp; Positive Health Factors
            </h3>
            {result.positive_factors && result.positive_factors.length > 0 ? (
              <ul className="rc-list">
                {result.positive_factors.map((f, i) => (
                  <li key={i} className="rc-item rc-item--pos">{f}</li>
                ))}
              </ul>
            ) : (
              <p className="rc-empty">No strong protective factors detected.</p>
            )}
          </div>

          {/* risk factors */}
          {result.risk_factors && result.risk_factors.length > 0 && (
            <div className="rc-detail-card">
              <h3 className="rc-detail-heading rc-detail-heading--risk">
                🔴&nbsp; Risk Factors
              </h3>
              <ul className="rc-list">
                {result.risk_factors.map((f, i) => (
                  <li key={i} className="rc-item rc-item--risk">{f}</li>
                ))}
              </ul>
            </div>
          )}

          {/* recommendations */}
          {result.recommendations && result.recommendations.length > 0 && (
            <div className="rc-detail-card rc-detail-card--rec">
              <h3 className="rc-detail-heading rc-detail-heading--rec">
                💡&nbsp; Recommendations
              </h3>
              <ul className="rc-list">
                {result.recommendations.map((r, i) => (
                  <li key={i} className="rc-item rc-item--rec">
                    <span className="rc-check">✔</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* disclaimer */}
        <p className="rc-disclaimer">
          ⚕️ This is an AI-assisted screening tool — <strong>not a medical diagnosis</strong>.
          Always consult a qualified healthcare professional.
        </p>
      </section>

      <style>{`
        /* ===== RESULT CARD ===== */
        .rc-section {
          max-width: 820px;
          margin: 0 auto;
          padding: 0 28px 72px;
          animation: rcFadeIn 0.5s ease-out;
        }
        @keyframes rcFadeIn {
          from { opacity: 0; transform: translateY(24px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        /* main card */
        .rc-card {
          position: relative;
          border: 1.5px solid;
          border-radius: 18px;
          padding: 36px 32px 28px;
          text-align: center;
          backdrop-filter: blur(10px);
        }
        .rc-badge {
          position: absolute;
          top: -18px; left: 50%;
          transform: translateX(-50%);
          width: 36px; height: 36px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1rem;
          box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        }
        .rc-risk {
          font-size: 1.7rem;
          font-weight: 800;
          margin: 8px 0 20px;
        }

        /* probability bar */
        .rc-bar-wrap {
          display: flex;
          align-items: center;
          gap: 14px;
          max-width: 480px;
          margin: 0 auto 16px;
        }
        .rc-bar-track {
          flex: 1;
          height: 12px;
          border-radius: 6px;
          background: rgba(255,255,255,0.08);
          overflow: hidden;
        }
        .rc-bar-fill {
          height: 100%;
          border-radius: 6px;
          transition: width 0.8s cubic-bezier(0.23,1,0.32,1);
        }
        .rc-bar-label {
          font-size: 1.15rem;
          font-weight: 800;
          min-width: 56px;
          text-align: right;
        }
        .rc-prob-text {
          color: #a1a0b3;
          font-size: 0.9rem;
          margin: 0;
        }

        /* detail cards */
        .rc-details {
          display: grid;
          gap: 16px;
          margin-top: 20px;
        }
        .rc-detail-card {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          padding: 22px 24px;
        }
        .rc-detail-heading {
          font-size: 0.95rem;
          font-weight: 700;
          margin: 0 0 14px;
        }
        .rc-detail-heading--pos  { color: #2cb67d; }
        .rc-detail-heading--risk { color: #ff6b6b; }
        .rc-detail-heading--rec  { color: #a78bfa; }

        .rc-list {
          list-style: none;
          padding: 0;
          margin: 0;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .rc-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 10px;
          font-size: 0.9rem;
          font-weight: 500;
        }
        .rc-item--pos {
          background: rgba(44,182,125,0.08);
          border: 1px solid rgba(44,182,125,0.2);
          color: #a8e6a3;
        }
        .rc-item--pos::before { content: '🟢'; margin-right: 2px; font-size: 0.8rem; }
        .rc-item--risk {
          background: rgba(255,69,58,0.08);
          border: 1px solid rgba(255,69,58,0.2);
          color: #ffacac;
        }
        .rc-item--risk::before { content: '🔴'; margin-right: 2px; font-size: 0.8rem; }
        .rc-item--rec {
          background: rgba(127,90,240,0.08);
          border: 1px solid rgba(127,90,240,0.18);
          color: #c5c3d4;
        }
        .rc-check {
          color: #7f5af0;
          font-size: 1rem;
          flex-shrink: 0;
        }

        .rc-empty {
          font-size: 0.88rem;
          color: #706f8a;
          font-style: italic;
          margin: 0;
        }

        /* disclaimer */
        .rc-disclaimer {
          margin-top: 24px;
          text-align: center;
          font-size: 0.82rem;
          color: #706f8a;
          line-height: 1.5;
        }

        /* responsive */
        @media (max-width: 600px) {
          .rc-card { padding: 28px 18px 22px; }
          .rc-risk { font-size: 1.3rem; }
        }
      `}</style>
    </>
  )
}
