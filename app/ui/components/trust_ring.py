def render_trust_ring(probability, size=100, stroke=8):
    """Generates an SVG trust ring gauge."""
    # Ensure probability is a float
    try:
        prob = float(probability)
    except (ValueError, TypeError):
        prob = 0.5

    prob = max(0.0, min(1.0, prob))
    percentage = int(prob * 100)

    # Determine color based on risk levels
    if prob < 0.3:
        color = "var(--low)"
    elif prob < 0.6:
        color = "var(--high)" # Matching user prompt's logic if applicable, but usually moderate
    elif prob < 0.85:
        color = "var(--high)"
    else:
        color = "var(--critical)"

    # Standard risk bands from prompt:
    # Low 0.0–0.3, Moderate 0.3–0.6, High 0.6–0.85, Critical 0.85–1.0
    if prob < 0.3:
        color = "var(--low)"
    elif prob < 0.6:
        color = "var(--high)" # Prompt says high-soft etc but let's map colors
        color = "#D97706" # High in prompt
    elif prob < 0.85:
        color = "#D97706"
    else:
        color = "var(--critical)"

    # Better color mapping from section 2 tokens
    if prob < 0.3: color = "var(--low)"
    elif prob < 0.6: color = "#D97706" # Moderate is missing a dedicated color in section 2 but High is D97706
    elif prob < 0.85: color = "var(--high)"
    else: color = "var(--critical)"

    radius = (size - stroke) / 2
    circumference = 2 * 3.14159 * radius
    offset = circumference - (prob * circumference)

    svg = f"""
    <div style="display: flex; align-items: center; justify-content: center; position: relative; width: {size}px; height: {size}px;">
        <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                fill="transparent"
                stroke="var(--surface-alt)"
                stroke-width="{stroke}"
            />
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                fill="transparent"
                stroke="{color}"
                stroke-width="{stroke}"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                stroke-linecap="round"
                style="transition: stroke-dashoffset 0.5s ease;"
            />
        </svg>
        <div style="position: absolute; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: {size/4}px; color: var(--text);">
            {percentage}%
        </div>
    </div>
    """
    return svg
