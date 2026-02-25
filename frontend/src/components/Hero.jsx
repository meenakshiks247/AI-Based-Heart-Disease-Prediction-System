import React from 'react'

export default function Hero({ onStartCheck }) {
  return (
    <>
      <section className="hf-hero" id="hero">
        {/* Decorative blobs */}
        <div className="hf-hero__blob hf-hero__blob--1" />
        <div className="hf-hero__blob hf-hero__blob--2" />

        <div className="hf-hero__content">
          <span className="hf-hero__badge">AI-Powered Cardiac Risk Analysis</span>
          <h1 className="hf-hero__title">
            AI Based <span className="hf-hero__title--accent">Heart Disease</span><br />Prediction System
          </h1>
          <p className="hf-hero__sub">
            Harness clinical-grade machine learning to evaluate your cardiac risk
            in seconds. Two analysis modes — clinical diagnosis and general health
            screening — provide actionable insights for a healthier heart.
          </p>
          <button className="hf-hero__cta" onClick={onStartCheck}>
            Start Health Check&nbsp;&nbsp;↓
          </button>
        </div>
      </section>

      {/* Info Sections */}
      <section className="hf-info" id="about">
        <div className="hf-info__inner">
          <h2 className="hf-info__heading">About Heart Disease</h2>
          <p className="hf-info__text">
            Heart disease is the <strong>leading cause of death worldwide</strong>, accounting for
            roughly 17.9 million lives each year. It encompasses conditions such as coronary
            artery disease, heart failure, arrhythmias, and valvular heart disease. Early
            detection through risk factor analysis can dramatically improve outcomes.
          </p>
          <p className="hf-info__text">
            Our AI models analyse your clinical parameters against patterns learned from
            thousands of real patient records, enabling fast, data-driven risk stratification.
          </p>
        </div>
      </section>

      <section className="hf-info hf-info--alt" id="symptoms">
        <div className="hf-info__inner">
          <h2 className="hf-info__heading">Common Symptoms</h2>
          <div className="hf-info__grid">
            {[
              { icon: '💔', title: 'Chest Pain', desc: 'Pressure, squeezing, or aching in the chest, often triggered by activity.' },
              { icon: '😮‍💨', title: 'Shortness of Breath', desc: 'Difficulty breathing during exertion or even at rest.' },
              { icon: '🫨', title: 'Palpitations', desc: 'Irregular, rapid, or pounding heartbeat sensations.' },
              { icon: '🥱', title: 'Fatigue', desc: 'Unusual tiredness, especially during everyday activities.' },
              { icon: '🤢', title: 'Nausea / Dizziness', desc: 'Lightheadedness, nausea, or cold sweats without obvious cause.' },
              { icon: '🦵', title: 'Swelling', desc: 'Swelling in the legs, ankles, or feet due to fluid retention.' },
            ].map((s) => (
              <div key={s.title} className="hf-info__card">
                <span className="hf-info__card-icon">{s.icon}</span>
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="hf-info" id="prevention">
        <div className="hf-info__inner">
          <h2 className="hf-info__heading">Prevention Tips</h2>
          <div className="hf-info__grid hf-info__grid--3">
            {[
              { icon: '🥗', title: 'Eat Healthy', desc: 'Choose fruits, vegetables, whole grains, lean protein, and healthy fats.' },
              { icon: '🏃', title: 'Stay Active', desc: 'Aim for at least 150 minutes of moderate aerobic activity per week.' },
              { icon: '🚭', title: 'Quit Smoking', desc: 'Smoking is one of the top risk factors — quitting halves your risk within a year.' },
              { icon: '⚖️', title: 'Maintain Weight', desc: 'Achieving and keeping a healthy BMI reduces cardiac and metabolic risk.' },
              { icon: '🧘', title: 'Manage Stress', desc: 'Chronic stress contributes to high blood pressure. Practice mindfulness.' },
              { icon: '🩺', title: 'Regular Check-ups', desc: 'Monitor blood pressure, cholesterol, and blood sugar routinely.' },
            ].map((s) => (
              <div key={s.title} className="hf-info__card">
                <span className="hf-info__card-icon">{s.icon}</span>
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <style>{`
        /* ======= HERO ======= */
        .hf-hero {
          position: relative;
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          background: linear-gradient(160deg, #0a0a1e 0%, #141432 50%, #1a1042 100%);
          padding: 80px 28px 40px;
        }
        .hf-hero__blob {
          position: absolute;
          border-radius: 50%;
          filter: blur(100px);
          opacity: 0.35;
          pointer-events: none;
        }
        .hf-hero__blob--1 {
          width: 500px; height: 500px;
          background: #7f5af0;
          top: -120px; left: -80px;
        }
        .hf-hero__blob--2 {
          width: 400px; height: 400px;
          background: #ff6b6b;
          bottom: -100px; right: -60px;
        }
        .hf-hero__content {
          position: relative;
          max-width: 740px;
          text-align: center;
          animation: heroFadeIn 0.9s ease-out;
        }
        @keyframes heroFadeIn {
          from { opacity: 0; transform: translateY(30px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .hf-hero__badge {
          display: inline-block;
          padding: 6px 16px;
          border-radius: 20px;
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 1.2px;
          background: rgba(127,90,240,0.2);
          color: #a78bfa;
          border: 1px solid rgba(127,90,240,0.3);
          margin-bottom: 20px;
        }
        .hf-hero__title {
          font-family: 'Inter', system-ui, -apple-system, sans-serif;
          font-size: 72px;
          font-weight: 800;
          line-height: 1.08;
          letter-spacing: -0.02em;
          color: #fff;
          margin: 0 0 24px;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
        .hf-hero__title--accent {
          background: linear-gradient(135deg, #ff6b6b, #7f5af0);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .hf-hero__sub {
          font-family: 'Inter', system-ui, -apple-system, sans-serif;
          color: #f1f5f9;
          font-size: 22px;
          font-weight: 500;
          line-height: 1.4;
          margin: 0 auto 36px;
          max-width: 60ch;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
        .hf-hero__cta {
          padding: 16px 42px;
          border: none;
          border-radius: 12px;
          font-size: 1.12rem;
          font-weight: 700;
          color: #fff;
          background: linear-gradient(135deg, #7f5af0, #2cb67d);
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
          font-family: inherit;
        }
        .hf-hero__cta:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 30px rgba(127,90,240,0.4);
        }

        /* ======= INFO SECTIONS ======= */
        .hf-info {
          padding: 72px 28px;
          background: #0e0e24;
        }
        .hf-info--alt { background: #111130; }
        .hf-info__inner {
          max-width: 960px;
          margin: 0 auto;
        }
        .hf-info__heading {
          font-size: 1.8rem;
          font-weight: 700;
          color: #fff;
          margin: 0 0 18px;
          text-align: center;
        }
        .hf-info__text {
          color: #a1a0b3;
          line-height: 1.7;
          font-size: 1rem;
          text-align: center;
          margin: 0 0 14px;
        }
        .hf-info__grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 20px;
          margin-top: 28px;
        }
        .hf-info__card {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 14px;
          padding: 24px 20px;
          transition: transform 0.25s, box-shadow 0.25s;
          text-align: center;
        }
        .hf-info__card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 28px rgba(127,90,240,0.15);
        }
        .hf-info__card-icon { font-size: 2rem; display: block; margin-bottom: 10px; }
        .hf-info__card h3 { color: #fff; font-size: 1rem; margin: 0 0 6px; }
        .hf-info__card p  { color: #908fa5; font-size: 0.88rem; line-height: 1.55; margin: 0; }

        /* lg ≤1024 — slightly reduce hero padding */
        @media (max-width: 1024px) {
          .hf-hero { padding: 96px 28px 56px; }
        }
        /* md ≤768 — tablet */
        @media (max-width: 768px) {
          .hf-hero { padding: 112px 24px 56px; }
          .hf-hero__title { font-size: 56px; }
          .hf-hero__sub  { font-size: 20px; }
          .hf-hero__stats { gap: 22px; }
          .hf-info__grid { grid-template-columns: 1fr 1fr; }
        }
        /* sm ≤480 — mobile */
        @media (max-width: 480px) {
          .hf-hero { padding: 96px 18px 44px; }
          .hf-hero__title { font-size: 48px; }
          .hf-hero__sub  { font-size: 18px; }
          .hf-hero__cta  { font-size: 1rem; padding: 14px 32px; }
          .hf-info__grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </>
  )
}
