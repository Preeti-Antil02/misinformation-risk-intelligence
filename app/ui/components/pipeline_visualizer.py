"""
app/ui/components/pipeline_visualizer.py
=========================================
Interactive, Animated Pipeline Visualization for RiskLens.
All node positions are calculated on a strict column grid so nothing overlaps.
Inspector panel is below SVG — never overlaps canvas.
All text is plain English (no code / formulas).
"""


def get_pipeline_html(theme: str = "dark") -> str:
    is_dark = (theme == "dark")
    bg = "#05060B" if is_dark else "#F8F9FD"
    surface = "rgba(13, 16, 28, 0.95)" if is_dark else "rgba(255, 255, 255, 0.95)"
    surface_alt = "rgba(20, 24, 42, 0.85)" if is_dark else "rgba(240, 243, 252, 0.9)"
    node_bg = "#080C1A" if is_dark else "#FFFFFF"
    node_stroke = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(203, 213, 225, 0.9)"
    node_hover_bg = "#10173A" if is_dark else "#F1F5F9"
    node_hover_stroke = "#818CF8" if is_dark else "#6366F1"
    edge_stroke = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(148, 163, 184, 0.45)"
    edge_marker = "rgba(255, 255, 255, 0.2)" if is_dark else "rgba(148, 163, 184, 0.7)"
    tc = "#F8FAFC" if is_dark else "#0F172A"
    muted = "#94A3B8" if is_dark else "#475569"
    border = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(203, 213, 225, 0.8)"
    stage_fill = "rgba(255, 255, 255, 0.015)" if is_dark else "rgba(99, 102, 241, 0.025)"
    stage_stroke = "rgba(255, 255, 255, 0.04)" if is_dark else "rgba(203, 213, 225, 0.5)"
    shadow_card = "0 8px 32px rgba(0, 0, 0, 0.4)" if is_dark else "0 8px 24px rgba(15, 23, 42, 0.08)"
    shadow_glow = "drop-shadow(0 0 9px rgba(129, 140, 248, 0.44))" if is_dark else "drop-shadow(0 4px 12px rgba(99, 102, 241, 0.2))"
    active_glow = "drop-shadow(0 0 15px rgba(56, 189, 248, 0.78))" if is_dark else "drop-shadow(0 4px 16px rgba(99, 102, 241, 0.28))"

    css_vars = f"""
    :root {{
        --bg:{bg}; --surface:{surface}; --surface-alt:{surface_alt}; --text:{tc}; --muted:{muted}; --border:{border};
        --pri:{'#4F46E5' if is_dark else '#6366F1'}; --pur:{'#7C3AED' if is_dark else '#4F46E5'}; --teal:#06B6D4;
        --em:#059669; --amb:#D97706; --rose:#DC2626;
        --fh:'Space Grotesk',sans-serif;
        --fb:'Plus Jakarta Sans',sans-serif;
        --fm:'JetBrains Mono',monospace;
    }}
    """

    # ──────────────────────────────────────────────────────────────
    # SVG Layout constants (viewBox 1480 × 880)
    # Stage columns (x-start, width):
    #   S1  12,  130  → right=142
    #   S2  155, 148  → right=303
    #   S3  316, 148  → right=464
    #   S4  477, 202  → right=679
    #   S5  692, 196  → right=888
    #   S6  901, 202  → right=1103
    #   S7  1116,148  → right=1264
    #   S8  1277,183  → right=1460
    # All nodes keep right-edge ≤ stage right edge.
    # ──────────────────────────────────────────────────────────────

    html = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>RiskLens — How It Works</title>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500"
        "&family=Plus+Jakarta+Sans:wght@400;500;600;700"
        "&family=Space+Grotesk:wght@400;500;700&display=swap' rel='stylesheet'>"
        "<style>" + css_vars + f"""
        *{{box-sizing:border-box;margin:0;padding:0}}
        html,body{{background:var(--bg);color:var(--text);font-family:var(--fb);min-height:100vh}}
        body{{padding:12px 14px 28px;overflow-x:hidden}}

        /* TOOLBAR */
        .tb{{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;
            gap:8px;background:var(--surface);border:1px solid var(--border);
            border-radius:12px;padding:11px 14px;margin-bottom:12px;
            box-shadow:{shadow_card};backdrop-filter:blur(14px)}}
        .tl{{font-family:var(--fh);font-size:11px;font-weight:700;
            color:var(--muted);letter-spacing:.05em;text-transform:uppercase;margin-right:4px}}
        .btn{{background:var(--surface-alt);border:1px solid var(--border);
             color:var(--text);font-family:var(--fb);font-size:12px;font-weight:600;
             padding:6px 14px;border-radius:8px;cursor:pointer;
             transition:all .18s;display:inline-flex;align-items:center;gap:5px}}
        .btn:hover{{background:rgba(99,102,241,.12);border-color:var(--pri);
                   transform:translateY(-1px);color:var(--pri)}}
        .btn.active{{background:linear-gradient(135deg,#4F46E5,#7C3AED);
                    border-color:transparent;color:#fff !important;
                    box-shadow:0 0 16px rgba(124,58,237,.42)}}
        .trbtn{{background:linear-gradient(135deg,#4F46E5,#7C3AED);color:#fff !important;
               border:none;padding:7px 16px;font-size:13px;font-weight:700;
               border-radius:8px;cursor:pointer;display:inline-flex;align-items:center;
               gap:6px;box-shadow:0 0 18px rgba(124,58,237,.38);
               transition:all .18s;font-family:var(--fb)}}
        .trbtn:hover{{transform:scale(1.04);box-shadow:0 0 28px rgba(124,58,237,.58)}}
        .sep{{width:1px;height:22px;background:var(--border);margin:0 3px}}
        .pill{{display:inline-flex;align-items:center;gap:6px;font-size:11px;
              color:var(--muted);background:var(--surface-alt);
              padding:5px 11px;border-radius:18px;border:1px solid var(--border)}}
        .dot{{width:7px;height:7px;border-radius:50%;background:#059669;
             box-shadow:0 0 8px #059669;animation:blink 2s infinite;flex-shrink:0}}
        @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.28}}}}

        /* CANVAS */
        .cw{{background:var(--surface);border:1px solid var(--border);
            border-radius:14px 14px 0 0;overflow:hidden;
            box-shadow:{shadow_card};position:relative}}
        svg#psvg{{width:100%;display:block}}

        /* SVG text */
        .slabel{{font-family:var(--fh);font-size:9px;font-weight:700;
                fill:var(--muted);letter-spacing:.07em;text-transform:uppercase;opacity:.85}}
        .ntag{{font-family:var(--fm);font-size:7.5px;font-weight:700;
              letter-spacing:.04em;fill:#38BDF8}}
        .ntag.p{{fill:#A78BFA}} .ntag.a{{fill:#F59E0B}} .ntag.r{{fill:#F87171}}
        .ntitle{{font-family:var(--fh);font-size:11px;font-weight:700;fill:var(--text)}}
        .nsub{{font-family:var(--fb);font-size:8.5px;fill:var(--muted)}}

        /* NODES */
        .node{{cursor:pointer}}
        .node .bg{{rx:9;fill:{node_bg};stroke:{node_stroke};
                  stroke-width:1.4;transition:all .22s}}
        .node:hover .bg,.node.hl .bg{{stroke:{node_hover_stroke};fill:{node_hover_bg};
            filter:{shadow_glow}}}
        .node.active .bg{{stroke:var(--pri);fill:{node_hover_bg};
            animation:np 1.5s ease-in-out infinite alternate}}
        @keyframes np{{from{{stroke-width:1.4}}
            to{{stroke-width:3.2;filter:{active_glow}}}}}

        /* EDGES */
        .edge{{fill:none;stroke:{edge_stroke};stroke-width:1.8;
              stroke-linecap:round}}
        .edge.lit{{stroke:var(--pri);stroke-width:2.4;
                  filter:drop-shadow(0 0 4px var(--pri))}}
        .edge.loop{{stroke:#D97706;stroke-width:2;stroke-dasharray:6 4;
                   animation:da 15s linear infinite}}
        .edge.loop.lit{{filter:drop-shadow(0 0 6px #D97706)}}
        @keyframes da{{to{{stroke-dashoffset:-380}}}}

        /* HINT */
        .hint{{position:absolute;top:50%;left:50%;
              transform:translate(-50%,-50%);
              background:var(--surface-alt);border:1px solid var(--border);
              padding:9px 18px;border-radius:9px;font-size:12px;
              color:var(--muted);pointer-events:none;
              backdrop-filter:blur(6px);transition:opacity .4s;white-space:nowrap}}
        .hint.gone{{opacity:0}}

        /* INSPECTOR (below SVG, no overlap) */
        .inspector{{background:var(--surface);
                   border:1px solid var(--border);
                   border-top:3px solid var(--pri);
                   border-radius:0 0 14px 14px;
                   box-shadow:{shadow_card};
                   min-height:172px;overflow:hidden}}
        .irow{{display:grid;grid-template-columns:84px 1fr 1fr 1fr}}
        .icol{{padding:16px 18px;border-right:1px solid var(--border)}}
        .icol:last-child{{border-right:none}}
        .iicon{{background:linear-gradient(135deg,rgba(79,70,229,.15),rgba(124,58,237,.08));
               display:flex;flex-direction:column;align-items:center;
               justify-content:center;gap:6px}}
        .big{{font-size:34px;line-height:1}}
        .istep{{font-family:var(--fm);font-size:8px;font-weight:700;
               color:var(--pri);text-align:center;letter-spacing:.04em;text-transform:uppercase}}
        .lbl{{font-family:var(--fh);font-size:8.5px;font-weight:700;
             letter-spacing:.07em;text-transform:uppercase;color:var(--pri);
             margin-bottom:5px;display:flex;align-items:center;gap:4px}}
        .ititle{{font-family:var(--fh);font-size:14px;font-weight:700;
                color:var(--text);margin-bottom:5px;line-height:1.25}}
        .ibadge{{display:inline-flex;font-family:var(--fm);font-size:8px;
                padding:3px 8px;border-radius:5px;font-weight:700;
                background:rgba(79,70,229,.12);color:var(--pri);
                border:1px solid rgba(79,70,229,.3);margin-bottom:9px}}
        .itext{{font-size:12px;color:var(--text);line-height:1.58}}
        .itext strong{{color:var(--text)}}
        .iobox{{background:var(--surface-alt);border:1px solid var(--border);
               border-radius:7px;padding:9px 11px;margin-top:8px}}
        .ior{{display:flex;align-items:flex-start;gap:7px;font-size:11.5px;
             color:var(--muted);line-height:1.5;margin-bottom:3px}}
        .ior:last-child{{margin-bottom:0}}
        .arr{{color:var(--pri);font-weight:700;flex-shrink:0}}
        .ival{{color:var(--text)}}
        .anabox{{background:rgba(6,182,212,.08);
                border:1px solid rgba(6,182,212,.25);
                border-radius:7px;padding:9px 11px;margin-top:8px}}
        .anatag{{font-family:var(--fm);font-size:8px;font-weight:700;
                color:#06B6D4;letter-spacing:.04em;margin-bottom:3px}}
        .anatext{{font-size:11.5px;color:var(--text);line-height:1.55;font-style:italic}}

        /* loop caption */
        .lc{{background:rgba(217,119,6,.1);border:1px solid rgba(217,119,6,.28);
            border-radius:10px;padding:8px 14px;margin-top:12px;font-size:12px;
            color:var(--text);display:flex;align-items:center;gap:8px}}
        .lc strong{{color:#F59E0B}}

        @media(max-width:800px){{
            .irow{{grid-template-columns:1fr}}
            .icol{{border-right:none;border-bottom:1px solid var(--border)}}
            .icol:last-child{{border-bottom:none}}
        }}
        @media(prefers-reduced-motion:reduce){{
            .edge.loop,.dot,.node.active .bg{{animation:none!important}}
        }}
        """ + "</style></head><body>"

        # HEADER
        "<div style='text-align:center;padding:18px 0 14px'>"
        "<h1 style='font-family:Space Grotesk,sans-serif;font-size:clamp(17px,2.6vw,24px);"
        "font-weight:700;color:var(--text);margin-bottom:5px'>🔍 How RiskLens Checks a Message</h1>"
        "<p style='font-size:12.5px;color:var(--muted)'>Click any block to learn what it does — "
        "no technical knowledge needed.</p></div>"

        # TOOLBAR
        "<div class='tb'>"
        "<span class='tl'>Show journey for:</span>"
        "<button class='btn active' id='btn-text' onclick='setSc(\"text\")'>💬 Text Message</button>"
        "<button class='btn' id='btn-url'  onclick='setSc(\"url\")'>🌐 News Link</button>"
        "<button class='btn' id='btn-image' onclick='setSc(\"image\")'>📸 Screenshot</button>"
        "<button class='btn' id='btn-retrain' onclick='setSc(\"retrain\")'>🔄 Nightly Learning</button>"
        "<div class='sep'></div>"
        "<button class='trbtn' id='play-btn' onclick='togglePlay()'>"
        "<span id='pico'>▶</span><span id='ptxt'> Step Through</span></button>"
        "<div class='pill'><div class='dot' id='sdot'></div>"
        "<span id='stxt'>Click a block or press Step Through</span></div>"
        "</div>"

        # CANVAS
        "<div class='cw'>"
        "<svg id='psvg' viewBox='0 0 1480 880'>"
        "<defs>"
        f"<marker id='ar'  viewBox='0 0 10 10' refX='7' refY='5' markerWidth='5' markerHeight='5' orient='auto-start-reverse'><path d='M0 2 L8 5 L0 8z' fill='{edge_marker}'/></marker>"
        "<marker id='arl' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='5' markerHeight='5' orient='auto-start-reverse'><path d='M0 2 L8 5 L0 8z' fill='#6366F1'/></marker>"
        "<marker id='ara' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='5' markerHeight='5' orient='auto-start-reverse'><path d='M0 2 L8 5 L0 8z' fill='#D97706'/></marker>"
        "<linearGradient id='gb' x1='0%' y1='0%' x2='100%' y2='100%'>"
        "<stop offset='0%' stop-color='#38BDF8'/><stop offset='100%' stop-color='#4F46E5'/></linearGradient>"
        "</defs>"

        # Stage bands
        f"<rect x='12'   y='22' width='130' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='22' y='42'>1 · Receive</text>"
        f"<rect x='155'  y='22' width='148' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='165' y='42'>2 · Read Image</text>"
        f"<rect x='316'  y='22' width='148' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='326' y='42'>3 · Language</text>"
        f"<rect x='477'  y='22' width='202' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='487' y='42'>4 · AI Detectors</text>"
        f"<rect x='692'  y='22' width='196' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='702' y='42'>5 · Score</text>"
        f"<rect x='901'  y='22' width='202' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='911' y='42'>6 · Fact-Check</text>"
        f"<rect x='1116' y='22' width='148' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='1126' y='42'>7 · Explain</text>"
        f"<rect x='1277' y='22' width='183' height='835' rx='10' fill='{stage_fill}' stroke='{stage_stroke}' stroke-width='1'/>"
        "<text class='slabel' x='1287' y='42'>8 · Learn</text>"

        # ─── EDGES ───────────────────────────────────────────────
        # S1→S3 text path
        "<path class='edge' id='e-in-text' d='M 132 224 L 320 228' marker-end='url(#ar)'/>"
        # S1→S5 URL path (arc over top)
        "<path class='edge' id='e-in-url'  d='M 132 200 C 300 200 500 115 696 115' marker-end='url(#ar)'/>"
        # S1→S2 image path
        "<path class='edge' id='e-in-ocr'  d='M 132 248 C 152 248 159 380 159 432' marker-end='url(#ar)'/>"
        # S2 internal
        "<path class='edge' id='e-ocr12'   d='M 229 504 L 229 537' marker-end='url(#ar)'/>"
        # S2→S3 OCR text to lang detect
        "<path class='edge' id='e-ocr-lang' d='M 293 576 C 312 576 316 400 316 274' marker-end='url(#ar)'/>"
        # S3→S4 EN branch
        "<path class='edge' id='e-lang-en'  d='M 452 206 L 481 188' marker-end='url(#ar)'/>"
        # S3→S4 Indic branch (long drop)
        "<path class='edge' id='e-lang-ind' d='M 452 255 C 469 255 481 660 481 689' marker-end='url(#ar)'/>"
        # S4 feature → models (vertical trunk)
        "<path class='edge' id='e-fb-lr'   d='M 578 220 L 578 252' marker-end='url(#ar)'/>"
        "<path class='edge' id='e-lr-xgb'  d='M 578 317 L 578 349' marker-end='url(#ar)'/>"
        "<path class='edge' id='e-xgb-rob' d='M 578 414 L 578 446' marker-end='url(#ar)'/>"
        "<path class='edge' id='e-rob-qwen' d='M 578 511 L 578 543' marker-end='url(#ar)'/>"
        # S4 models → S5 stack (fan right into stack)
        "<path class='edge' id='e-lr-stack'   d='M 679 284 C 691 284 696 370 696 392' marker-end='url(#ar)'/>"
        "<path class='edge' id='e-xgb-stack'  d='M 679 381 C 691 381 696 398 696 400' marker-end='url(#ar)'/>"
        "<path class='edge' id='e-rob-stack'  d='M 679 478 C 691 478 696 415 696 415' marker-end='url(#ar)'/>"
        "<path class='edge' id='e-qwen-stack' d='M 679 575 C 691 575 696 430 696 430' marker-end='url(#ar)'/>"
        # S5 stack → score (vertical)
        "<path class='edge' id='e-stack-score' d='M 787 440 L 787 490' marker-end='url(#ar)'/>"
        # S5 URL credibility → score (vertical from same stage top)
        "<path class='edge' id='e-url-score'  d='M 787 168 L 787 490' marker-end='url(#ar)'/>"
        # S4 MuRIL → S5 score (Indic bypass)
        "<path class='edge' id='e-muril-score' d='M 679 689 C 691 689 696 590 696 595' marker-end='url(#ar)'/>"
        # S5 score → S6 ag1 (right and up)
        "<path class='edge' id='e-score-ag1'  d='M 888 542 C 898 542 905 350 905 260' marker-end='url(#ar)'/>"
        # S6 agent chain
        "<path class='edge' id='e-ag12'        d='M 1001 260 L 1001 368' marker-end='url(#ar)'/>"
        "<path class='edge' id='e-ag23'        d='M 1001 480 L 1001 596' marker-end='url(#ar)'/>"
        # S6 ag3 → S7 shap
        "<path class='edge' id='e-ag-shap'    d='M 1103 638 C 1113 638 1120 500 1120 414' marker-end='url(#ar)'/>"
        # S7 shap → reply (vertical)
        "<path class='edge' id='e-shap-reply' d='M 1186 416 L 1186 490' marker-end='url(#ar)'/>"
        # S7 reply → S8 feedback
        "<path class='edge' id='e-reply-fb'   d='M 1264 536 L 1281 548' marker-end='url(#ar)'/>"
        # S8 feedback → queue (vertical)
        "<path class='edge' id='e-fb-queue'   d='M 1357 586 L 1357 645' marker-end='url(#ar)'/>"

        # NIGHTLY LOOP (amber dashed) — queue bottom back to stack
        "<path class='edge loop' id='e-loop1' d='M 1357 743 C 1357 820 900 820 700 820' marker-end='url(#ara)'/>"
        "<path class='edge loop' id='e-loop2' d='M 700 820 C 500 820 490 750 490 724' marker-end='url(#ara)'/>"
        "<path class='edge loop' id='e-loop3' d='M 490 724 C 490 680 600 440 696 394' marker-end='url(#ara)'/>"
        # Loop caption
        "<text x='550' y='840' font-family='Plus Jakarta Sans,sans-serif' font-size='9' fill='#F59E0B' opacity='.72'>"
        "↺  Every night at 2AM: 500+ user corrections collected → AI re-trains → smarter model goes live instantly</text>"

        # ─── NODES ───────────────────────────────────────────────
        # S1 — Receive
        "<g class='node' id='n-ingest' onclick='ins(\"ingest\")'>"
        "<rect class='bg' x='18' y='178' width='114' height='90'/>"
        "<text class='ntag' x='28' y='196'>STEP 1</text>"
        "<text class='ntitle' x='28' y='213'>📩 Receive</text>"
        "<text class='nsub' x='28' y='229'>Message via</text>"
        "<text class='nsub' x='28' y='242'>Telegram</text>"
        "</g>"

        # S2a — Clean image
        "<g class='node' id='n-ocr-val' onclick='ins(\"ocr_val\")'>"
        "<rect class='bg' x='159' y='432' width='133' height='72'/>"
        "<text class='ntag p' x='168' y='450'>STEP 2A</text>"
        "<text class='ntitle' x='168' y='467'>🧹 Clean Image</text>"
        "<text class='nsub' x='168' y='482'>Sharpen • denoise</text>"
        "<text class='nsub' x='168' y='494'>• threshold</text>"
        "</g>"

        # S2b — Read text from image
        "<g class='node' id='n-ocr-ext' onclick='ins(\"ocr_ext\")'>"
        "<rect class='bg' x='159' y='537' width='133' height='80'/>"
        "<text class='ntag p' x='168' y='554'>STEP 2B</text>"
        "<text class='ntitle' x='168' y='570'>👁 Read Text</text>"
        "<text class='nsub' x='168' y='585'>from image</text>"
        "<text class='nsub' x='168' y='598'>2 OCR engines</text>"
        "<text class='nsub' x='168' y='609'>3 attempts each</text>"
        "</g>"

        # S3 — Language detect
        "<g class='node' id='n-lang' onclick='ins(\"lang\")'>"
        "<rect class='bg' x='320' y='182' width='136' height='96'/>"
        "<text class='ntag' x='330' y='200'>STEP 3</text>"
        "<text class='ntitle' x='330' y='217'>🌐 Detect</text>"
        "<text class='ntitle' x='330' y='232'>Language</text>"
        "<text class='nsub' x='330' y='250'>English / Hindi</text>"
        "<text class='nsub' x='330' y='263'>Tamil / Bengali...</text>"
        "</g>"

        # S4 — Feature builder (top)
        "<g class='node' id='n-feat' onclick='ins(\"feat\")'>"
        "<rect class='bg' x='481' y='148' width='190' height='72'/>"
        "<text class='ntag' x='491' y='166'>STEP 4A</text>"
        "<text class='ntitle' x='491' y='183'>📊 Find Warning Signs</text>"
        "<text class='nsub' x='491' y='199'>11 manipulation signals</text>"
        "<text class='nsub' x='491' y='211'>caps · emojis · urgency</text>"
        "</g>"

        # S4 models stacked vertically
        "<g class='node' id='n-lr' onclick='ins(\"lr\")'>"
        "<rect class='bg' x='481' y='252' width='190' height='65'/>"
        "<text class='ntag' x='491' y='269'>AI DETECTOR 1</text>"
        "<text class='ntitle' x='491' y='286'>📝 Word Pattern AI</text>"
        "<text class='nsub' x='491' y='302'>Checks writing style</text>"
        "</g>"

        "<g class='node' id='n-xgb' onclick='ins(\"xgb\")'>"
        "<rect class='bg' x='481' y='349' width='190' height='65'/>"
        "<text class='ntag' x='491' y='366'>AI DETECTOR 2</text>"
        "<text class='ntitle' x='491' y='383'>🌲 Pattern Trees AI</text>"
        "<text class='nsub' x='491' y='399'>Spots deceptive combos</text>"
        "</g>"

        "<g class='node' id='n-rob' onclick='ins(\"rob\")'>"
        "<rect class='bg' x='481' y='446' width='190' height='65'/>"
        "<text class='ntag p' x='491' y='463'>AI DETECTOR 3</text>"
        "<text class='ntitle' x='491' y='480'>🧠 Deep Context AI</text>"
        "<text class='nsub' x='491' y='496'>Reads full meaning</text>"
        "</g>"

        "<g class='node' id='n-qwen' onclick='ins(\"qwen\")'>"
        "<rect class='bg' x='481' y='543' width='190' height='65'/>"
        "<text class='ntag p' x='491' y='560'>AI DETECTOR 4</text>"
        "<text class='ntitle' x='491' y='577'>💬 Reasoning AI</text>"
        "<text class='nsub' x='491' y='593'>Thinks step-by-step</text>"
        "</g>"

        # S4 MuRIL (bottom — Indic)
        "<g class='node' id='n-muril' onclick='ins(\"muril\")'>"
        "<rect class='bg' x='481' y='652' width='190' height='72'/>"
        "<text class='ntag a' x='491' y='669'>🇮🇳 INDIC SPECIALIST</text>"
        "<text class='ntitle' x='491' y='686'>Indian Language AI</text>"
        "<text class='nsub' x='491' y='701'>Hindi · Tamil · Telugu</text>"
        "<text class='nsub' x='491' y='713'>Bengali · Gujarati...</text>"
        "</g>"

        # S5 — Source credibility (top of score column)
        "<g class='node' id='n-cred' onclick='ins(\"cred\")'>"
        "<rect class='bg' x='696' y='96' width='183' height='72'/>"
        "<text class='ntag' x='706' y='115'>SOURCE CHECK</text>"
        "<text class='ntitle' x='706' y='132'>🏛 Website Trusted?</text>"
        "<text class='nsub' x='706' y='149'>100+ outlet database</text>"
        "<text class='nsub' x='706' y='161'>MBFC · OpenSources</text>"
        "</g>"

        # S5 — Stack ensemble
        "<g class='node' id='n-stack' onclick='ins(\"stack\")'>"
        "<rect class='bg' x='696' y='350' width='183' height='90'/>"
        "<text class='ntag' x='706' y='368'>COMBINE VOTES</text>"
        "<text class='ntitle' x='706' y='385'>🗳 All 4 AIs Vote</text>"
        "<text class='nsub' x='706' y='402'>Weighted majority rule</text>"
        "<text class='nsub' x='706' y='415'>Meta-AI picks best combo</text>"
        "<text class='nsub' x='706' y='428'>5-fold cross-validation</text>"
        "</g>"

        # S5 — Integrated risk score (bottom)
        "<g class='node' id='n-score' onclick='ins(\"score\")'>"
        "<rect class='bg' x='696' y='490' width='183' height='110'/>"
        "<text class='ntag r' x='706' y='508'>FINAL RISK SCORE</text>"
        "<text class='ntitle' x='706' y='525'>🎯 Calculate</text>"
        "<text class='ntitle' x='706' y='540'>Risk Level</text>"
        "<text class='nsub' x='706' y='558'>🟢 Low · 🟡 Medium</text>"
        "<text class='nsub' x='706' y='571'>🔴 High · 🚨 Critical</text>"
        "<text class='nsub' x='706' y='584'>Platt calibrated</text>"
        "</g>"

        # S6 — Agent 1
        "<g class='node' id='n-ag1' onclick='ins(\"ag1\")'>"
        "<rect class='bg' x='905' y='186' width='190' height='74'/>"
        "<text class='ntag' x='915' y='204'>FACT-CHECK 1</text>"
        "<text class='ntitle' x='915' y='222'>🔎 Pick the Main</text>"
        "<text class='ntitle' x='915' y='237'>Claim to Check</text>"
        "<text class='nsub' x='915' y='252'>Extract verifiable statement</text>"
        "</g>"

        # S6 — Agent 2
        "<g class='node' id='n-ag2' onclick='ins(\"ag2\")'>"
        "<rect class='bg' x='905' y='368' width='190' height='112'/>"
        "<text class='ntag p' x='915' y='386'>FACT-CHECK 2</text>"
        "<text class='ntitle' x='915' y='403'>🌍 Search the Web</text>"
        "<text class='nsub' x='915' y='420'>Google Fact Check API</text>"
        "<text class='nsub' x='915' y='434'>Known debunks library</text>"
        "<text class='nsub' x='915' y='448'>Live DuckDuckGo search</text>"
        "<text class='nsub' x='915' y='462'>BeautifulSoup extraction</text>"
        "</g>"

        # S6 — Agent 3
        "<g class='node' id='n-ag3' onclick='ins(\"ag3\")'>"
        "<rect class='bg' x='905' y='596' width='190' height='86'/>"
        "<text class='ntag a' x='915' y='614'>FACT-CHECK 3</text>"
        "<text class='ntitle' x='915' y='631'>⚖️ Weigh Evidence</text>"
        "<text class='ntitle' x='915' y='647'>&amp; Write Verdict</text>"
        "<text class='nsub' x='915' y='662'>FALSE/TRUE/MISLEADING</text>"
        "</g>"

        # S7 — SHAP
        "<g class='node' id='n-shap' onclick='ins(\"shap\")'>"
        "<rect class='bg' x='1120' y='338' width='136' height='78'/>"
        "<text class='ntag' x='1130' y='355'>WHY IT FLAGGED</text>"
        "<text class='ntitle' x='1130' y='372'>🔬 Explain the</text>"
        "<text class='ntitle' x='1130' y='387'>Red Flags</text>"
        "<text class='nsub' x='1130' y='403'>Top 3 signals</text>"
        "</g>"

        # S7 — Reply card
        "<g class='node' id='n-reply' onclick='ins(\"reply\")'>"
        "<rect class='bg' x='1120' y='490' width='136' height='95'/>"
        "<text class='ntag r' x='1130' y='508'>STEP 7</text>"
        "<text class='ntitle' x='1130' y='525'>📤 Send Your</text>"
        "<text class='ntitle' x='1130' y='540'>Result Card</text>"
        "<text class='nsub' x='1130' y='557'>Risk score + verdict</text>"
        "<text class='nsub' x='1130' y='569'>👍 Correct · 👎 Wrong</text>"
        "</g>"

        # S8 — Feedback DB
        "<g class='node' id='n-fb' onclick='ins(\"fb\")'>"
        "<rect class='bg' x='1281' y='510' width='170' height='76'/>"
        "<text class='ntag' x='1291' y='528'>YOUR FEEDBACK</text>"
        "<text class='ntitle' x='1291' y='545'>📝 Record Vote</text>"
        "<text class='nsub' x='1291' y='562'>Saved to database</text>"
        "<text class='nsub' x='1291' y='575'>Priority scored</text>"
        "</g>"

        # S8 — Retrain queue
        "<g class='node' id='n-queue' onclick='ins(\"queue\")'>"
        "<rect class='bg' x='1281' y='645' width='170' height='98'/>"
        "<text class='ntag a' x='1291' y='663'>🌙 EVERY NIGHT 2AM</text>"
        "<text class='ntitle' x='1291' y='680'>📈 Re-Train AI</text>"
        "<text class='nsub' x='1291' y='697'>500+ corrections needed</text>"
        "<text class='nsub' x='1291' y='710'>F1 test gate → live</text>"
        "<text class='nsub' x='1291' y='723'>Overwrites both models</text>"
        "</g>"

        "</svg>"
        "<div class='hint' id='hint'>👆 Click any glowing block to learn what it does</div>"
        "</div>"

        # ─── INSPECTOR PANEL (below SVG — zero overlap) ───────────
        "<div class='inspector'>"
        "<div class='irow'>"
        "<div class='icol iicon'>"
        "<div class='big' id='i-emo'>📩</div>"
        "<div class='istep' id='i-step'>STEP 1 OF 8</div>"
        "</div>"
        "<div class='icol'>"
        "<div class='lbl'>📌 What is this?</div>"
        "<div class='ititle' id='i-title'>Receiving Your Message</div>"
        "<div class='ibadge' id='i-badge'>ENTRY POINT</div>"
        "<div class='itext' id='i-what'>"
        "This is the <strong>front door</strong> of RiskLens — a Telegram chatbot that receives "
        "whatever you send and immediately routes it to the checking pipeline."
        "</div>"
        "</div>"
        "<div class='icol'>"
        "<div class='lbl'>💡 Simple Analogy</div>"
        "<div class='anabox'>"
        "<div class='anatag'>THINK OF IT AS...</div>"
        "<div class='anatext' id='i-ana'>"
        "A postbox at the entrance of a fact-checking newsroom. You drop your letter in "
        "and the team inside picks it up and starts working on it immediately."
        "</div></div>"
        "</div>"
        "<div class='icol'>"
        "<div class='lbl'>📥 Input → 📤 Output</div>"
        "<div class='iobox'>"
        "<div class='ior'><span class='arr'>IN</span>"
        "<span class='ival' id='i-in'>Your message, link, or image</span></div>"
        "<div class='ior'><span class='arr'>OUT</span>"
        "<span class='ival' id='i-out'>Text content ready for analysis</span></div>"
        "</div>"
        "</div>"
        "</div>"
        "</div>"

        # LOOP CAPTION
        "<div class='lc'>🔄 <strong>The amber dashed arrows</strong> show the nightly learning loop — "
        "user votes are collected all day, then every night the AI re-trains on those corrections "
        "and becomes smarter by morning.</div>"

        # ─── JS ────────────────────────────────────────────────────
        "<script>" +
        """
var ND = {
  ingest:{emo:"📩",step:"STEP 1 OF 8",title:"Receiving Your Message",badge:"ENTRY POINT",
    what:"This is the <strong>front door</strong> of RiskLens. A Telegram chatbot receives whatever you send — plain text, a news link, or a screenshot — and immediately passes it to the fact-checking pipeline.",
    ana:"A postbox at the entrance of a fact-checking newsroom. You drop your letter in and the team picks it up and starts working immediately.",
    inp:"Your message, link, or image sent via Telegram",
    out:"Raw text (or image file) handed off to the next step"},
  ocr_val:{emo:"🧹",step:"STEP 2A OF 8",title:"Cleaning the Image",badge:"IMAGE PREP",
    what:"Before reading text from a screenshot, RiskLens <strong>cleans and sharpens the image</strong>. It removes noise, evens lighting, and makes text edges crisp so the reading step is far more accurate.",
    ana:"Like using a bright lamp and a magnifying glass before reading a blurry photocopy — the cleaner the page, the fewer reading mistakes.",
    inp:"Raw screenshot or photo you sent",
    out:"A cleaned, sharpened image ready for text reading"},
  ocr_ext:{emo:"👁",step:"STEP 2B OF 8",title:"Reading Text from the Image",badge:"DUAL OCR",
    what:"RiskLens uses <strong>two AI reading engines</strong>. If the first misses text, the second tries. If both fail on the processed version, they each try on the original image — giving up to 4 chances to get the text right.",
    ana:"Two editors read the same blurry newspaper. If neither can make out a word, they try again under better lighting. They only give up after 4 attempts.",
    inp:"Cleaned image file",
    out:"Extracted text with a confidence score (e.g. 87% confident)"},
  lang:{emo:"🌐",step:"STEP 3 OF 8",title:"Detecting the Language",badge:"LANGUAGE ROUTER",
    what:"RiskLens automatically detects the language — English, Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi and more. It then <strong>routes the message to the right specialist AI</strong> for that language.",
    ana:"A receptionist at a multilingual help desk. When you walk in, they instantly detect your language and direct you to the right expert's desk.",
    inp:"Your text (any language or script)",
    out:"Language identified → routed to the correct AI detector"},
  feat:{emo:"📊",step:"STEP 4A OF 8",title:"Spotting Warning Signs in the Writing",badge:"11 SIGNALS",
    what:"Before the AI detectors run, RiskLens measures <strong>11 warning signals</strong> in the writing — ALL-CAPS WORDS, excessive exclamation marks!!!, emotional urgency words, missing sources, digit overload. These are known tricks used in viral misinformation.",
    ana:"A librarian who checks a book cover before reading it: Is the title all caps? Too many exclamation points? No author or publisher listed? These are red flags before opening a single page.",
    inp:"Your text message (any language)",
    out:"11 warning scores (e.g. caps_ratio=0.45, exclamations=4)"},
  lr:{emo:"📝",step:"STEP 4B OF 8",title:"AI Detector 1 — Word Patterns",badge:"LINEAR CLASSIFIER",
    what:"The first AI compares your text's <strong>vocabulary patterns</strong> to tens of thousands of real and fake news articles it was trained on. It asks: do the specific words used tend to appear more in misinformation or trustworthy reporting?",
    ana:"A detective who has read 50,000 articles and learned that phrases like 'doctors don't want you to know' almost always appear in fake news — not in reputable journalism.",
    inp:"Your message broken into words",
    out:"Suspicion score: e.g. 52% suspicious"},
  xgb:{emo:"🌲",step:"STEP 4C OF 8",title:"AI Detector 2 — Combination Patterns",badge:"GRADIENT BOOSTING",
    what:"The second AI checks <strong>hundreds of combinations</strong> of warning signs together, not just individual words. It looks at how writing style, punctuation, and vocabulary combine in suspicious ways.",
    ana:"A polygraph expert who doesn't look at one signal but hundreds simultaneously: sweating AND raised heart rate AND voice pitch — a combination that matches known deception patterns.",
    inp:"Your message + 11 warning signal scores",
    out:"Suspicion score: e.g. 87% suspicious"},
  rob:{emo:"🧠",step:"STEP 4D OF 8",title:"AI Detector 3 — Deep Understanding",badge:"LANGUAGE MODEL",
    what:"The third AI (RoBERTa, 125 million parameters) <strong>reads your full message in context</strong>. It understands meaning, tone, implied claims, and logical structure — not just word lists.",
    ana:"Asking a PhD linguist to read your message. They don't just look for suspicious words — they understand what is actually being claimed and whether the reasoning holds together.",
    inp:"Full text of your message",
    out:"Suspicion score: e.g. 81% suspicious"},
  qwen:{emo:"💬",step:"STEP 4E OF 8",title:"AI Detector 4 — Step-by-Step Reasoning",badge:"REASONING MODEL",
    what:"The fourth AI (Qwen2.5, 3 billion parameters) <strong>thinks out loud step by step</strong>: Is this verifiable? Does it use emotional manipulation? Are sources missing? It scores based on that chain of reasoning.",
    ana:"A skeptical journalist thinking aloud: 'This claims scientists discovered X, but no study is cited, and the language is designed to cause panic — that is a red flag.'",
    inp:"Your message + a structured thinking prompt",
    out:"Suspicion score with a brief reasoning trace"},
  muril:{emo:"🇮🇳",step:"STEP 4F OF 8",title:"Indian Language Specialist AI",badge:"INDIC EXPERT",
    what:"If your message is in Hindi, Tamil, Telugu, Bengali, Marathi, or Gujarati, it goes to a <strong>specialist AI trained only on Indian languages</strong> (MuRIL). It bypasses all English AI detectors and achieves perfect accuracy on Hindi test data.",
    ana:"Calling a regional specialist who grew up reading local newspapers and knows exactly what misinformation looks like in that community — rather than a generalist trained only on English content.",
    inp:"Your message in an Indian language",
    out:"Suspicion score from an India-specific expert AI"},
  cred:{emo:"🏛",step:"STEP 5A OF 8",title:"Is This Website Trustworthy?",badge:"SOURCE DATABASE",
    what:"If you shared a link, RiskLens checks the <strong>website's reputation</strong> in a database of 100+ news sources rated by media bias organisations. Known satire sites, conspiracy outlets, and tabloids are flagged. Reputable outlets improve the score.",
    ana:"Asking a press freedom watchdog: Is this outlet registered? Do they publish corrections? Have they been caught fabricating stories? Those answers change how much you trust the article.",
    inp:"The website URL extracted from your link",
    out:"Trustworthiness score 0–100% + bias label (e.g. Far-Right, Satire)"},
  stack:{emo:"🗳",step:"STEP 5B OF 8",title:"All 4 AIs Vote — Best Combination Wins",badge:"META-AI ENSEMBLE",
    what:"All 4 AI detectors give their individual scores. A <strong>meta-AI (trained with 5-fold cross-validation)</strong> then combines those scores, having learned which AI is most reliable in which situation — weighting their votes accordingly.",
    ana:"A jury of 4 expert witnesses. The judge doesn't just count votes — they know which expert is most credible for which type of evidence and weights their testimony accordingly.",
    inp:"Suspicion scores from all 4 AI detectors",
    out:"Single combined suspicion score (most accurate available)"},
  score:{emo:"🎯",step:"STEP 5C OF 8",title:"Final Risk Level",badge:"PLATT CALIBRATED",
    what:"The combined AI score is <strong>calibrated for accuracy</strong> (Platt Scaling ensures the percentages are honest, not overconfident). Then website trust is mixed in (25%) so a suspicious article on an untrustworthy site gets a higher rating.",
    ana:"A doctor's second opinion that adjusts the initial test result for known error rates, then combines it with the patient's history (the source reputation) to give a final, reliable diagnosis.",
    inp:"Combined AI score + website trust score (if URL present)",
    out:"Risk level: 🟢 Low · 🟡 Medium · 🔴 High · 🚨 Critical"},
  ag1:{emo:"🔎",step:"STEP 6A OF 8",title:"Pick the Key Claim to Check",badge:"CLAIM EXTRACTOR",
    what:"Your message may have many sentences. This step <strong>identifies the single most verifiable factual statement</strong> — the specific claim that can be checked against real-world evidence. Opinions and feelings are skipped.",
    ana:"A fact-checker's first step: they underline the specific checkable statement — 'Drug X cures cancer' — rather than trying to verify every word in the whole article.",
    inp:"Your full message",
    out:"The one key factual claim (e.g. 'Vaccine X causes autism')"},
  ag2:{emo:"🌍",step:"STEP 6B OF 8",title:"Search the Web for Evidence",badge:"WEB RESEARCHER",
    what:"RiskLens searches <strong>three sources in order</strong>: 1) Google's official Fact Check database, 2) a built-in library of 7 famous debunked viral hoaxes, 3) a live DuckDuckGo web search with article extraction.",
    ana:"A librarian who first checks the most authoritative encyclopedia, then the known-errors register, then does a fresh newspaper search — collecting all available evidence.",
    inp:"The key claim extracted in the previous step",
    out:"List of sources and evidence found (or a note if nothing was found)"},
  ag3:{emo:"⚖️",step:"STEP 6C OF 8",title:"Weigh Evidence and Write a Verdict",badge:"VERDICT SYNTHESIZER",
    what:"Evidence is <strong>weighed against the original claim</strong>. Web evidence counts for 70%, AI suspicion counts for 30%. If no evidence was found, the verdict says 'Inconclusive' — the system never fabricates a verdict.",
    ana:"A judge who considers both physical evidence (70%) and expert witness testimony (30%). If there is no physical evidence, the judge refuses to convict — the verdict is 'not proven.'",
    inp:"Evidence from web search + original AI suspicion score",
    out:"Verdict: FALSE / TRUE / MISLEADING / INCONCLUSIVE"},
  shap:{emo:"🔬",step:"STEP 7A OF 8",title:"Explain Why It Was Flagged",badge:"SHAP EXPLAINER",
    what:"After scoring, RiskLens <strong>explains which exact words triggered the suspicion</strong>. Using SHAP values, it identifies the 3 most suspicious signals and writes them in plain language so you can judge for yourself.",
    ana:"A doctor who not only says 'we think you have condition X' but also explains exactly which test results led to that conclusion — blood pressure, cholesterol, ECG — so you can form your own view.",
    inp:"Your text + the final risk score",
    out:"Top 3 red flags explained in plain English"},
  reply:{emo:"📤",step:"STEP 7B OF 8",title:"Sending You the Result",badge:"REPLY CARD",
    what:"Everything is formatted into a <strong>clear result card</strong> sent back on Telegram: risk score, verdict, website trust rating, the claim that was checked, sources found, and top suspicious phrases — all in one message.",
    ana:"A lab report from a clinic: not just a number, but a full summary with what was tested, what the results mean, and the doctor's recommendation — all on one clean, readable page.",
    inp:"All results from every step above",
    out:"Your risk report card with 👍 Correct / 👎 Wrong vote buttons"},
  fb:{emo:"📝",step:"STEP 8A OF 8",title:"Recording Your Vote",badge:"FEEDBACK LOGGER",
    what:"When you tap 👍 (correct) or 👎 (wrong), that vote is <strong>saved with the original prediction</strong>. Cases where the AI was most wrong are given the highest priority in the overnight retraining queue.",
    ana:"Filling out a patient satisfaction form — except these forms directly improve the diagnosis software. The more specific the correction, the bigger the improvement.",
    inp:"Your 👍 or 👎 tap",
    out:"Saved correction with priority score for overnight retraining"},
  queue:{emo:"📈",step:"STEP 8B OF 8",title:"Nightly AI Re-Training",badge:"F1 GATE",
    what:"Every night at 2AM, if 500+ user corrections were collected, RiskLens <strong>re-trains the AI models</strong> on those corrections. The new model only goes live if it scores better than the old one — it can never get worse.",
    ana:"A chess engine that reviews all the games it lost during the day, updates its strategy overnight, but only accepts the update if the new strategy wins more test matches than the old one.",
    inp:"500+ user corrections collected during the day",
    out:"Updated, smarter AI model live from next morning"}
};

var SC = {
  text:{col:"#38BDF8",lbl:"Journey: text message through all 4 AI detectors",
    nodes:["n-ingest","n-lang","n-feat","n-lr","n-xgb","n-rob","n-qwen","n-stack","n-score","n-ag1","n-ag2","n-ag3","n-shap","n-reply","n-fb"],
    edges:["e-in-text","e-lang-en","e-fb-lr","e-lr-xgb","e-xgb-rob","e-rob-qwen","e-lr-stack","e-xgb-stack","e-rob-stack","e-qwen-stack","e-stack-score","e-score-ag1","e-ag12","e-ag23","e-ag-shap","e-shap-reply","e-reply-fb"]},
  url:{col:"#06B6D4",lbl:"Journey: news link — website trust score mixes into risk rating",
    nodes:["n-ingest","n-cred","n-lang","n-feat","n-lr","n-xgb","n-stack","n-score","n-ag1","n-ag2","n-ag3","n-reply"],
    edges:["e-in-text","e-in-url","e-url-score","e-lang-en","e-fb-lr","e-lr-xgb","e-lr-stack","e-xgb-stack","e-stack-score","e-score-ag1","e-ag12","e-ag23","e-ag-shap","e-shap-reply"]},
  image:{col:"#A78BFA",lbl:"Journey: screenshot → clean → read → Indian language AI",
    nodes:["n-ingest","n-ocr-val","n-ocr-ext","n-lang","n-muril","n-score","n-ag1","n-ag2","n-ag3","n-shap","n-reply","n-fb"],
    edges:["e-in-ocr","e-ocr12","e-ocr-lang","e-lang-ind","e-muril-score","e-score-ag1","e-ag12","e-ag23","e-ag-shap","e-shap-reply","e-reply-fb"]},
  retrain:{col:"#F59E0B",lbl:"Nightly loop: corrections → retrain → F1 gate → live model",
    nodes:["n-reply","n-fb","n-queue","n-muril","n-stack"],
    edges:["e-reply-fb","e-fb-queue","e-loop1","e-loop2","e-loop3"]}
};

var cur="text", playing=false, timer=null, idx=0;

function hideHint(){
  var h=document.getElementById("hint");
  if(h){h.classList.add("gone");setTimeout(function(){h.style.display="none"},450);}
}

function ins(k){
  hideHint();
  var d=ND[k]; if(!d) return;
  document.getElementById("i-emo").textContent=d.emo;
  document.getElementById("i-step").textContent=d.step;
  document.getElementById("i-title").textContent=d.title;
  document.getElementById("i-badge").textContent=d.badge;
  document.getElementById("i-what").innerHTML=d.what;
  document.getElementById("i-ana").textContent=d.ana;
  document.getElementById("i-in").textContent=d.inp;
  document.getElementById("i-out").textContent=d.out;
  document.querySelectorAll(".node").forEach(function(n){n.classList.remove("hl")});
  var el=document.getElementById("n-"+k.replace(/_/g,"-"));
  if(el) el.classList.add("hl");
}

function setSc(s){
  cur=s; playing=false;
  if(timer) clearTimeout(timer);
  document.getElementById("pico").textContent="▶";
  document.getElementById("ptxt").textContent=" Step Through";
  document.querySelectorAll(".tb .btn").forEach(function(b){b.classList.remove("active")});
  document.getElementById("btn-"+s).classList.add("active");
  var sc=SC[s];
  document.getElementById("sdot").style.background=sc.col;
  document.getElementById("sdot").style.boxShadow="0 0 10px "+sc.col;
  document.getElementById("stxt").textContent=sc.lbl;
  document.querySelectorAll(".node").forEach(function(n){n.classList.remove("active","hl")});
  document.querySelectorAll(".edge").forEach(function(e){e.classList.remove("lit")});
  sc.nodes.forEach(function(id){var n=document.getElementById(id);if(n)n.classList.add("active");});
  sc.edges.forEach(function(id){var e=document.getElementById(id);if(e)e.classList.add("lit");});
  idx=0;
}

function togglePlay(){
  if(playing){
    playing=false; if(timer)clearTimeout(timer);
    document.getElementById("pico").textContent="▶";
    document.getElementById("ptxt").textContent=" Step Through";
  }else{
    playing=true;
    document.getElementById("pico").textContent="❚❚";
    document.getElementById("ptxt").textContent=" Pause";
    step();
  }
}

function step(){
  if(!playing) return;
  var sc=SC[cur];
  if(idx>=sc.nodes.length) idx=0;
  var nid=sc.nodes[idx];
  document.querySelectorAll(".node").forEach(function(n){n.classList.remove("active")});
  var el=document.getElementById(nid);
  if(el){
    el.classList.add("active");
    ins(nid.replace("n-","").replace(/-/g,"_"));
  }
  idx++;
  timer=setTimeout(step,1200);
}

window.addEventListener("DOMContentLoaded",function(){setSc("text");ins("ingest");});
""" +
        "</script></body></html>"
    )
    return html
