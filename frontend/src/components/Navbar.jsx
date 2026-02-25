import React, { useState, useEffect } from 'react'

export default function Navbar({ onPredictClick }) {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const links = [
    { label: 'Home', href: '#hero' },
    { label: 'About Heart Disease', href: '#about' },
    { label: 'Symptoms', href: '#symptoms' },
    { label: 'Prevention', href: '#prevention' },
    { label: 'Try Prediction', href: '#predict' },
  ]

  const scrollTo = (href) => {
    setMobileOpen(false)
    const el = document.querySelector(href)
    if (el) el.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <>
      <nav className={`hf-nav ${scrolled ? 'hf-nav--scrolled' : ''}`}>
        <div className="hf-nav__inner">
          {/* Logo */}
          <a className="hf-nav__logo" href="#hero" onClick={(e) => { e.preventDefault(); scrollTo('#hero') }}>
            <span className="hf-nav__heart">❤️</span>
            <span className="hf-nav__brand">Heartify AI</span>
          </a>

          {/* Desktop links */}
          <ul className="hf-nav__links">
            {links.map((l) => (
              <li key={l.href}>
                <a href={l.href} onClick={(e) => { e.preventDefault(); scrollTo(l.href) }}>
                  {l.label}
                </a>
              </li>
            ))}
          </ul>

          {/* CTA */}
          <button className="hf-nav__cta" onClick={onPredictClick}>
            Predict Now
          </button>

          {/* Hamburger */}
          <button
            className={`hf-nav__burger ${mobileOpen ? 'hf-nav__burger--open' : ''}`}
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle menu"
          >
            <span /><span /><span />
          </button>
        </div>

        {/* Mobile drawer */}
        {mobileOpen && (
          <div className="hf-nav__mobile">
            {links.map((l) => (
              <a key={l.href} href={l.href} onClick={(e) => { e.preventDefault(); scrollTo(l.href) }}>
                {l.label}
              </a>
            ))}
            <button className="hf-nav__cta hf-nav__cta--mobile" onClick={() => { setMobileOpen(false); onPredictClick() }}>
              Predict Now
            </button>
          </div>
        )}
      </nav>

      <style>{`
        .hf-nav {
          position: fixed;
          top: 0; left: 0; right: 0;
          z-index: 1000;
          transition: background 0.3s, box-shadow 0.3s, backdrop-filter 0.3s;
          background: transparent;
        }
        .hf-nav--scrolled {
          background: rgba(10, 10, 30, 0.85);
          backdrop-filter: blur(16px);
          box-shadow: 0 2px 24px rgba(0,0,0,0.4);
        }
        .hf-nav__inner {
          max-width: 1200px;
          margin: 0 auto;
          display: flex;
          align-items: center;
          padding: 14px 28px;
          gap: 8px;
        }
        .hf-nav__logo {
          display: flex;
          align-items: center;
          gap: 8px;
          text-decoration: none;
          margin-right: auto;
        }
        .hf-nav__heart { font-size: 1.5rem; }
        .hf-nav__brand {
          font-size: 1.25rem;
          font-weight: 800;
          background: linear-gradient(135deg, #ff6b6b, #7f5af0);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          letter-spacing: -0.3px;
        }
        .hf-nav__links {
          list-style: none;
          display: flex;
          gap: 6px;
        }
        .hf-nav__links a {
          text-decoration: none;
          color: #b0aec1;
          font-size: 0.85rem;
          font-weight: 500;
          padding: 6px 12px;
          border-radius: 8px;
          transition: color 0.2s, background 0.2s;
        }
        .hf-nav__links a:hover {
          color: #fff;
          background: rgba(127, 90, 240, 0.15);
        }
        .hf-nav__cta {
          padding: 8px 20px;
          border: none;
          border-radius: 8px;
          font-size: 0.85rem;
          font-weight: 700;
          color: #fff;
          background: linear-gradient(135deg, #7f5af0, #2cb67d);
          cursor: pointer;
          transition: opacity 0.2s, transform 0.15s;
          font-family: inherit;
          margin-left: 8px;
        }
        .hf-nav__cta:hover { opacity: 0.9; transform: translateY(-1px); }

        /* hamburger */
        .hf-nav__burger {
          display: none;
          flex-direction: column;
          gap: 5px;
          background: none;
          border: none;
          cursor: pointer;
          padding: 4px;
          margin-left: 12px;
        }
        .hf-nav__burger span {
          display: block;
          width: 22px; height: 2px;
          background: #b0aec1;
          border-radius: 2px;
          transition: transform 0.3s, opacity 0.3s;
        }
        .hf-nav__burger--open span:nth-child(1) { transform: translateY(7px) rotate(45deg); }
        .hf-nav__burger--open span:nth-child(2) { opacity: 0; }
        .hf-nav__burger--open span:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }

        /* mobile drawer */
        .hf-nav__mobile {
          display: none;
          flex-direction: column;
          padding: 8px 28px 20px;
          gap: 4px;
          background: rgba(10,10,30,0.95);
        }
        .hf-nav__mobile a {
          text-decoration: none;
          color: #b0aec1;
          font-size: 0.95rem;
          padding: 10px 0;
          border-bottom: 1px solid rgba(255,255,255,0.06);
          transition: color 0.2s;
        }
        .hf-nav__mobile a:hover { color: #fff; }
        .hf-nav__cta--mobile { margin-top: 8px; width: 100%; }

        @media (max-width: 860px) {
          .hf-nav__links, .hf-nav__cta:not(.hf-nav__cta--mobile) { display: none; }
          .hf-nav__burger { display: flex; }
          .hf-nav__mobile { display: flex; }
        }
      `}</style>
    </>
  )
}
