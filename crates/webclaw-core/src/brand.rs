/// Brand identity extraction from HTML.
/// Pure DOM/CSS analysis: extracts colors, fonts, logo, and favicon
/// from style blocks, inline styles, and semantic HTML patterns.
/// No network calls, no LLM — WASM-safe.
use std::collections::HashMap;
use std::sync::LazyLock as Lazy;

use regex::Regex;
use scraper::{ElementRef, Html, Selector};
use serde::Serialize;
use url::Url;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct BrandColor {
    pub hex: String,
    pub usage: ColorUsage,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
pub enum ColorUsage {
    Primary,
    Secondary,
    Background,
    Text,
    Accent,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogoVariant {
    pub url: String,
    pub kind: String, // "favicon", "apple-touch-icon", "logo", "og-image", "svg"
}

#[derive(Debug, Clone, Serialize)]
pub struct BrandIdentity {
    /// Brand name extracted from og:site_name, application-name, or <title>.
    pub name: Option<String>,
    pub colors: Vec<BrandColor>,
    pub fonts: Vec<String>,
    /// All discovered logo variants (favicon, apple-touch-icon, og:image, etc.)
    pub logos: Vec<LogoVariant>,
    /// Primary logo URL (best guess).
    pub logo_url: Option<String>,
    pub favicon_url: Option<String>,
    /// Open Graph image (background/hero image).
    pub og_image: Option<String>,
}

// ---------------------------------------------------------------------------
// Regex patterns (compiled once)
// ---------------------------------------------------------------------------

/// Matches CSS declarations with a property and value, e.g. `color: #fff;`
static CSS_DECL: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)([\w-]+)\s*:\s*([^;}{]+)").unwrap());

/// Matches hex colors: #RGB or #RRGGBB
static HEX_COLOR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"#([0-9a-fA-F]{3})\b|#([0-9a-fA-F]{6})\b").unwrap());

/// Matches rgb(r, g, b)
static RGB_COLOR: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)").unwrap()
});

/// Matches rgba(r, g, b, a)
static RGBA_COLOR: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*[\d.]+\s*\)").unwrap()
});

/// Matches hsl(h, s%, l%)
static HSL_COLOR: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)hsla?\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*(?:,\s*[\d.]+\s*)?\)")
        .unwrap()
});

/// Matches the family tail of CSS `font:` shorthand after size/line-height.
static FONT_SHORTHAND_FAMILY: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?ix)
        (?:^|\s)
        (?:xx-small|x-small|small|medium|large|x-large|xx-large|larger|smaller|\d*\.?\d+(?:px|rem|em|pt|pc|in|cm|mm|%|vw|vh|vmin|vmax))
        (?:\s*/\s*[^\s,]+)?
        \s+
        (.+)$
    "#,
    )
    .unwrap()
});

/// Largest `NxN` token in a sizes attribute or URL (`32x32`, `180×180`).
/// `sizes="any"` does not match.
static ICON_DIMENSION: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)([0-9]{1,5})\s*[x×]\s*([0-9]{1,5})").unwrap());

/// `touch-76`, `icon-256`, `favicon-32`, `apple-touch-icon-180`.
static ICON_FILENAME_SIZE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:apple-touch-icon|favicon|touch|icon)[-_]?([0-9]{2,4})").unwrap()
});

const LOGO_EXCLUDE_TERMS: &[&str] = &[
    "customer",
    "partner",
    "client",
    "press-logo",
    "logo-bar",
    "customers",
];

macro_rules! selector {
    ($s:expr) => {{
        static SEL: Lazy<Selector> = Lazy::new(|| Selector::parse($s).unwrap());
        &*SEL
    }};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract brand identity from raw HTML.
///
/// `html` -- raw HTML string
/// `url`  -- optional base URL for resolving relative paths
pub fn extract_brand(html: &str, url: Option<&str>) -> BrandIdentity {
    let doc = Html::parse_document(html);
    let base_url = url.and_then(|u| Url::parse(u).ok());

    let name = extract_brand_name(&doc);
    let css_sources = collect_css(&doc);
    let colors = extract_colors(&css_sources, name.as_deref());
    let fonts = extract_fonts(&css_sources, name.as_deref());
    let mut logos = find_all_logos(&doc, base_url.as_ref());
    let best_icon = find_best_icon(&doc, base_url.as_ref());
    let mut favicon_url = find_favicon(&doc, base_url.as_ref());
    let og_image = find_og_image(&doc, base_url.as_ref());

    // Explicit wordmark (including SVG) wins. Otherwise a raster Organization
    // logo, then the best raster icon link, then a public favicon for the host.
    let mut logo_url = find_logo(&doc, base_url.as_ref());
    if logo_url.is_none() {
        logo_url = find_jsonld_logo(&doc, base_url.as_ref());
    }
    if logo_url.is_none()
        && let Some((icon_url, score)) = best_icon.as_ref()
        && *score > 1
    {
        logo_url = Some(icon_url.clone());
    }
    if logo_url.is_none()
        && let Some(fallback) = base_url.as_ref().and_then(public_favicon_url)
    {
        logo_url = Some(fallback.clone());
        if favicon_url.is_none() {
            favicon_url = Some(fallback.clone());
        }
        if logos.iter().all(|logo| logo.url != fallback) {
            logos.push(LogoVariant {
                url: fallback,
                kind: "favicon".to_string(),
            });
        }
    }

    BrandIdentity {
        name,
        colors,
        fonts,
        logos,
        logo_url,
        favicon_url,
        og_image,
    }
}

// ---------------------------------------------------------------------------
// CSS collection
// ---------------------------------------------------------------------------

/// A CSS declaration with its property context and raw value.
struct CssDecl {
    property: String,
    value: String,
}

/// Collect all CSS from <style> blocks, style="" attributes, CSS custom
/// properties, Tailwind arbitrary values, and meta theme-color.
fn collect_css(doc: &Html) -> Vec<CssDecl> {
    let mut decls = Vec::new();

    // <style> blocks
    for el in doc.select(selector!("style")) {
        let text: String = el.text().collect();
        parse_declarations(&text, &mut decls);
        // Also extract CSS custom properties (--brand-color: #xxx)
        parse_css_variables(&text, &mut decls);
    }

    // Inline style="" attributes on all elements
    for el in doc.select(selector!("[style]")) {
        if let Some(style) = el.value().attr("style") {
            parse_declarations(style, &mut decls);
            parse_css_variables(style, &mut decls);
        }
    }

    // Tailwind arbitrary color values from class attributes (bg-[#xxx], text-[#xxx])
    for el in doc.select(selector!("[class]")) {
        if let Some(class) = el.value().attr("class") {
            parse_tailwind_colors(class, &mut decls);
        }
    }

    // <meta name="theme-color"> — mobile browser chrome color
    for el in doc.select(selector!("meta[name='theme-color']")) {
        if let Some(content) = el.value().attr("content") {
            decls.push(CssDecl {
                property: "background-color".to_string(),
                value: content.to_string(),
            });
        }
    }

    // <link rel="preload" as="font"> — extract font family from href filename
    for el in doc.select(selector!("link[rel='preload'][as='font']")) {
        if let Some(href) = el.value().attr("href")
            && let Some(font_name) = extract_font_name_from_url(href)
        {
            decls.push(CssDecl {
                property: "font-family".to_string(),
                value: format!("\"{}\"", font_name),
            });
        }
    }

    // <link rel="stylesheet"> with Google Fonts — extract family from URL
    for el in doc.select(selector!("link[rel='stylesheet']")) {
        if let Some(href) = el.value().attr("href")
            && (href.contains("fonts.googleapis.com") || href.contains("fonts.bunny.net"))
        {
            for font in extract_google_fonts_from_url(href) {
                decls.push(CssDecl {
                    property: "font-family".to_string(),
                    value: format!("\"{}\"", font),
                });
            }
        }
    }

    decls
}

fn parse_declarations(css_text: &str, out: &mut Vec<CssDecl>) {
    for cap in CSS_DECL.captures_iter(css_text) {
        let property = cap[1].to_ascii_lowercase();
        let value = cap[2].trim().to_string();
        out.push(CssDecl { property, value });
    }
}

/// Extract CSS custom properties that look like color values.
/// e.g., `--primary: #3b82f6;` or `--brand-bg: rgb(30, 41, 59);`
static CSS_VAR: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)--([\w-]+)\s*:\s*([^;}{]+)").unwrap());

fn parse_css_variables(css_text: &str, out: &mut Vec<CssDecl>) {
    for cap in CSS_VAR.captures_iter(css_text) {
        let var_name = cap[1].to_ascii_lowercase();
        let value = cap[2].trim().to_string();

        // Only keep vars that contain a color value
        if HEX_COLOR.is_match(&value)
            || RGB_COLOR.is_match(&value)
            || RGBA_COLOR.is_match(&value)
            || HSL_COLOR.is_match(&value)
        {
            // Classify by variable name
            let property = if var_name.contains("background") || var_name.contains("bg") {
                "background-color"
            } else if var_name.contains("text")
                || var_name.contains("foreground")
                || var_name.contains("fg")
            {
                "color"
            } else if var_name.contains("border") || var_name.contains("accent") {
                "border-color"
            } else {
                "color"
            };
            out.push(CssDecl {
                property: property.to_string(),
                value,
            });
        }
    }
}

/// Extract Tailwind arbitrary color values from class strings.
/// e.g., `bg-[#1a1a2e]`, `text-[#e94560]`, `border-[rgb(255,0,0)]`
static TW_COLOR: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:bg|text|border|ring|outline|shadow|accent|fill|stroke)-\[([^\]]+)\]").unwrap()
});

fn parse_tailwind_colors(class: &str, out: &mut Vec<CssDecl>) {
    for cap in TW_COLOR.captures_iter(class) {
        let value = &cap[1];
        if HEX_COLOR.is_match(value)
            || RGB_COLOR.is_match(value)
            || RGBA_COLOR.is_match(value)
            || HSL_COLOR.is_match(value)
        {
            // Classify by prefix
            let full = cap.get(0).unwrap().as_str();
            let property = if full.starts_with("bg-") {
                "background-color"
            } else if full.starts_with("text-") {
                "color"
            } else if full.starts_with("border-") {
                "border-color"
            } else {
                "color"
            };
            out.push(CssDecl {
                property: property.to_string(),
                value: value.to_string(),
            });
        }
    }
}

/// Extract a font family name from a font file URL.
/// e.g., `/fonts/Inter-Variable.woff2` → "Inter"
fn extract_font_name_from_url(url: &str) -> Option<String> {
    let filename = url.rsplit('/').next()?;
    let stem = filename.split('.').next()?;
    // Clean up common suffixes: -Regular, -Bold, -Variable, -subset, etc.
    let clean = stem
        .split('-')
        .take_while(|part| {
            !matches!(
                part.to_lowercase().as_str(),
                "regular"
                    | "bold"
                    | "italic"
                    | "light"
                    | "medium"
                    | "semibold"
                    | "variable"
                    | "subset"
                    | "latin"
                    | "cyrillic"
                    | "woff"
                    | "woff2"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    if clean.is_empty() || clean.len() < 2 {
        None
    } else {
        Some(clean)
    }
}

/// Extract font family names from a Google Fonts CSS URL.
/// e.g., `https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Roboto+Mono`
fn extract_google_fonts_from_url(url: &str) -> Vec<String> {
    let mut fonts = Vec::new();
    for part in url.split('&') {
        let family = if let Some(rest) = part.strip_prefix("family=") {
            rest
        } else if let Some(rest) = part.split("family=").nth(1) {
            rest
        } else {
            continue;
        };
        // Take everything before `:` (weight/style specs)
        let name = family.split(':').next().unwrap_or(family);
        // Replace + with space
        let clean = name.replace('+', " ");
        if !clean.is_empty() {
            fonts.push(clean);
        }
    }
    fonts
}

// ---------------------------------------------------------------------------
// Color extraction
// ---------------------------------------------------------------------------

/// Colors to filter out: pure white, pure black, and common grays.
fn is_boring_color(hex: &str) -> bool {
    matches!(
        hex,
        "#FFFFFF"
            | "#000000"
            | "#F8F8F8"
            | "#F5F5F5"
            | "#EEEEEE"
            | "#E5E5E5"
            | "#DDDDDD"
            | "#D4D4D4"
            | "#CCCCCC"
            | "#BBBBBB"
            | "#AAAAAA"
            | "#999999"
            | "#888888"
            | "#777777"
            | "#666666"
            | "#555555"
            | "#444444"
            | "#333333"
            | "#222222"
            | "#111111"
            | "#F0F0F0"
            | "#E0E0E0"
            | "#D0D0D0"
            | "#C0C0C0"
            | "#B0B0B0"
            | "#A0A0A0"
            | "#909090"
            | "#808080"
            | "#707070"
            | "#606060"
            | "#505050"
            | "#404040"
            | "#303030"
            | "#202020"
            | "#101010"
            | "#FAFAFA"
            | "#F9F9F9"
            | "#F7F7F7"
            | "#F4F4F4"
            | "#F3F3F3"
            | "#F2F2F2"
            | "#F1F1F1"
            | "#EFEFEF"
            | "#EBEBEB"
            | "#E8E8E8"
    )
}

fn extract_colors(decls: &[CssDecl], brand_name: Option<&str>) -> Vec<BrandColor> {
    // Track (hex, usage) -> count
    let mut counts: HashMap<String, HashMap<ColorUsage, usize>> = HashMap::new();

    for decl in decls {
        let usage = classify_color_property(&decl.property);
        for hex in parse_colors_from_value(&decl.value) {
            if is_boring_color(&hex) {
                continue;
            }
            *counts
                .entry(hex)
                .or_default()
                .entry(usage.clone())
                .or_insert(0) += 1;
        }
    }

    // Flatten into BrandColor entries, picking the most common usage per color
    let mut colors: Vec<BrandColor> = counts
        .into_iter()
        .map(|(hex, usage_map)| {
            let total: usize = usage_map.values().sum();
            let usage = usage_map
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(u, _)| u)
                .unwrap_or(ColorUsage::Unknown);
            BrandColor {
                hex,
                usage,
                count: total,
            }
        })
        .collect();

    // Sort by frequency (descending)
    colors.sort_by_key(|c| std::cmp::Reverse(c.count));

    demote_or_remove_oauth_palette(&mut colors, brand_name);

    // Promote top non-white/black to Primary/Secondary if they're still Unknown
    let mut assigned_primary = colors.iter().any(|c| c.usage == ColorUsage::Primary);
    let mut assigned_secondary = colors.iter().any(|c| c.usage == ColorUsage::Secondary);

    for color in &mut colors {
        if color.usage != ColorUsage::Unknown {
            continue;
        }
        if !assigned_primary {
            color.usage = ColorUsage::Primary;
            assigned_primary = true;
        } else if !assigned_secondary {
            color.usage = ColorUsage::Secondary;
            assigned_secondary = true;
        }
    }

    colors.truncate(10);
    colors
}

const GOOGLE_OAUTH_COLORS: &[&str] = &[
    "#1A73E8", "#4285F4", "#34A853", "#FBBC05", "#EA4335", "#5F6368", "#202124", "#E8EAED",
    "#F1F3F4",
];

fn demote_or_remove_oauth_palette(colors: &mut Vec<BrandColor>, brand_name: Option<&str>) {
    let brand = brand_name.unwrap_or("").to_ascii_lowercase();
    if brand.contains("google") {
        return;
    }

    let google_hits = colors
        .iter()
        .filter(|c| GOOGLE_OAUTH_COLORS.contains(&c.hex.as_str()))
        .count();
    if google_hits < 3 {
        return;
    }

    colors.retain(|c| !GOOGLE_OAUTH_COLORS.contains(&c.hex.as_str()));
}

fn classify_color_property(property: &str) -> ColorUsage {
    match property {
        "background-color" | "background" => ColorUsage::Background,
        "color" => ColorUsage::Text,
        "border-color"
        | "border"
        | "border-top-color"
        | "border-bottom-color"
        | "border-left-color"
        | "border-right-color"
        | "outline-color" => ColorUsage::Accent,
        _ => ColorUsage::Unknown,
    }
}

/// Parse all color values from a single CSS value string.
fn parse_colors_from_value(value: &str) -> Vec<String> {
    let mut colors = Vec::new();

    // Hex colors
    for cap in HEX_COLOR.captures_iter(value) {
        if let Some(short) = cap.get(1) {
            colors.push(expand_short_hex(short.as_str()));
        } else if let Some(full) = cap.get(2) {
            colors.push(format!("#{}", full.as_str().to_ascii_uppercase()));
        }
    }

    // rgb()
    for cap in RGB_COLOR.captures_iter(value) {
        let r: u8 = cap[1].parse().unwrap_or(0);
        let g: u8 = cap[2].parse().unwrap_or(0);
        let b: u8 = cap[3].parse().unwrap_or(0);
        colors.push(format!("#{:02X}{:02X}{:02X}", r, g, b));
    }

    // rgba()
    for cap in RGBA_COLOR.captures_iter(value) {
        let r: u8 = cap[1].parse().unwrap_or(0);
        let g: u8 = cap[2].parse().unwrap_or(0);
        let b: u8 = cap[3].parse().unwrap_or(0);
        colors.push(format!("#{:02X}{:02X}{:02X}", r, g, b));
    }

    // hsl() / hsla()
    for cap in HSL_COLOR.captures_iter(value) {
        let h: f64 = cap[1].parse().unwrap_or(0.0);
        let s: f64 = cap[2].parse::<f64>().unwrap_or(0.0) / 100.0;
        let l: f64 = cap[3].parse::<f64>().unwrap_or(0.0) / 100.0;
        let (r, g, b) = hsl_to_rgb(h, s, l);
        colors.push(format!("#{:02X}{:02X}{:02X}", r, g, b));
    }

    colors
}

/// Expand 3-digit hex (#F00) to 6-digit (#FF0000).
fn expand_short_hex(short: &str) -> String {
    let chars: Vec<char> = short.chars().collect();
    format!(
        "#{}{}{}{}{}{}",
        chars[0], chars[0], chars[1], chars[1], chars[2], chars[2]
    )
    .to_ascii_uppercase()
}

/// Convert HSL to RGB. h in [0, 360], s and l in [0.0, 1.0].
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    if s == 0.0 {
        let v = (l * 255.0).round() as u8;
        return (v, v, v);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);

    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

// ---------------------------------------------------------------------------
// Font extraction
// ---------------------------------------------------------------------------

/// Generic font families that aren't brand-specific.
const GENERIC_FONTS: &[&str] = &[
    "serif",
    "sans-serif",
    "monospace",
    "cursive",
    "fantasy",
    "system-ui",
    "ui-serif",
    "ui-sans-serif",
    "ui-monospace",
    "ui-rounded",
    "emoji",
    "math",
    "fangsong",
    "inherit",
    "initial",
    "unset",
    "revert",
    "arial",
    "times",
    "times new roman",
    "courier new",
    "georgia",
    "menlo",
    "monaco",
    "consolas",
    "liberation mono",
    "sf mono",
    "sfmono-regular",
    "source code pro",
    "apple color emoji",
    "segoe ui",
    "segoe ui emoji",
    "segoe ui symbol",
    "noto color emoji",
    "blinkmacsystemfont",
    "-apple-system",
];

fn extract_fonts(decls: &[CssDecl], brand_name: Option<&str>) -> Vec<String> {
    let mut freq: HashMap<String, usize> = HashMap::new();
    let brand = brand_name.unwrap_or("").to_ascii_lowercase();

    for decl in decls {
        if decl.property != "font-family" && decl.property != "font" {
            continue;
        }

        // For shorthand `font:`, extract only the family tail after the
        // size/line-height token. The previous implementation treated values
        // like `500 12px Roboto` as a font family, which polluted `/v1/brand`
        // output with CSS declarations instead of usable family names.
        let family_str = if decl.property == "font" {
            match parse_font_shorthand_family(&decl.value) {
                Some(family) => family,
                None => continue,
            }
        } else {
            decl.value.clone()
        };

        for font in split_font_families(&family_str) {
            let lower = font.to_lowercase();
            if !GENERIC_FONTS.contains(&lower.as_str())
                && !is_junk_font_name(&lower)
                && !is_third_party_auth_font(&lower, &brand)
            {
                *freq.entry(font).or_insert(0) += 1;
            }
        }
    }

    let mut fonts: Vec<(String, usize)> = freq.into_iter().collect();
    fonts.sort_by_key(|f| std::cmp::Reverse(f.1));
    fonts.into_iter().map(|(name, _)| name).collect()
}

fn is_third_party_auth_font(name: &str, brand_name: &str) -> bool {
    !brand_name.contains("google") && name.contains("google sans")
}

fn parse_font_shorthand_family(value: &str) -> Option<String> {
    let caps = FONT_SHORTHAND_FAMILY.captures(value)?;
    let mut family = caps.get(1)?.as_str().trim().to_string();

    // Drop the optional slash line-height residue if it was not consumed due
    // to unusual whitespace, then leave comma-separated family names intact.
    if let Some(stripped) = family.strip_prefix('/') {
        family = stripped
            .split_once(' ')
            .map(|(_, rest)| rest)
            .unwrap_or("")
            .trim()
            .to_string();
    }

    if family.is_empty() {
        None
    } else {
        Some(family)
    }
}

/// Filter out junk font names: CSS variables, hex hashes (Next.js font optimization),
/// single-character names, and other non-human-readable values.
fn is_junk_font_name(name: &str) -> bool {
    // CSS variable references: var(--font-mono)
    if name.starts_with("var(") {
        return true;
    }
    // Hex hash strings (Next.js font optimization): "5b01f339abf2f1a5"
    if name.len() >= 8 && name.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    if name
        .split_whitespace()
        .next()
        .is_some_and(|part| part.len() >= 8 && part.chars().all(|c| c.is_ascii_hexdigit()))
    {
        return true;
    }
    // Too short to be a real font name
    if name.len() < 3 {
        return true;
    }
    // Third-party rendering libraries and icon fonts overwhelm app shells
    // like claude.com/openai.com but are not product typography.
    if name.contains("katex")
        || name.contains("open dyslexic")
        || name.contains("opendyslexic")
        || name.contains("math")
        || name.contains("fraktur")
        || name.contains("caligraphic")
        || name.contains("typewriter")
        || name.contains("glyph")
        || name.contains("icon")
        || name.contains("emoji")
        || name.contains("symbol")
    {
        return true;
    }
    // Malformed shorthand leftovers and CSS-internal values.
    if name.contains(')')
        || name.contains('!')
        || name.contains('/')
        || name.contains("px ")
        || name.contains("rem ")
        || name.contains("em ")
    {
        return true;
    }
    // Starts with underscore or double dash (CSS internals)
    if name.starts_with('_') || name.starts_with("--") {
        return true;
    }
    false
}

/// Split a font-family CSS value into individual font names.
/// Handles both quoted and unquoted names.
fn split_font_families(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| {
            s.trim()
                .trim_matches('"')
                .trim_matches('\'')
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Logo detection
// ---------------------------------------------------------------------------

fn find_logo(doc: &Html, base_url: Option<&Url>) -> Option<String> {
    for el in doc.select(selector!("header img, nav img")) {
        if let Some(url) = header_logo_url(el, base_url) {
            return Some(url);
        }
    }

    // Homepage link with an image. Skip customer/partner grids so a row of
    // client logos that all point at "/" cannot become the brand mark.
    for el in doc.select(selector!("a[href='/'] img, a[href] img")) {
        if image_has_excluded_logo_term(el) {
            continue;
        }
        if let Some(parent) = el.parent().and_then(|p| p.value().as_element()) {
            let href = parent.attr("href").unwrap_or("");
            if (href == "/" || href.ends_with(".com") || href.ends_with(".com/"))
                && let Some(src) = el.value().attr("src")
            {
                return Some(resolve_url(src, base_url));
            }
        }
    }

    find_single_unambiguous_logo(doc, base_url)
}

/// Header/nav logo. `class`, `id`, `alt`, `src`, `aria-label`, and the first
/// srcset URL are all logo signals. A customer/partner/press grid is not.
fn header_logo_url(el: ElementRef<'_>, base_url: Option<&Url>) -> Option<String> {
    let class = el.value().attr("class").unwrap_or("");
    let id = el.value().attr("id").unwrap_or("");
    let alt = el.value().attr("alt").unwrap_or("");
    let aria = el.value().attr("aria-label").unwrap_or("");
    let src = el.value().attr("src").unwrap_or("").trim();
    let srcset_attr = el.value().attr("srcset").unwrap_or("");
    let srcset = first_srcset_candidate(srcset_attr);
    let fields = [class, id, alt, aria, src, srcset];
    if fields.iter().copied().any(contains_excluded_logo_term) {
        return None;
    }
    let attr_signal = [class, id, alt, aria]
        .iter()
        .any(|field| contains_ci(field, "logo"));
    let src_signal = contains_ci(src, "logo");
    let srcset_signal = contains_ci(srcset, "logo");
    if !attr_signal && !src_signal && !srcset_signal {
        return None;
    }

    if !src.is_empty() && (attr_signal || src_signal) {
        return Some(resolve_url(src, base_url));
    }
    if !srcset.is_empty() {
        return Some(resolve_url(srcset, base_url));
    }
    None
}

/// One `<img>` anywhere whose class/id/alt/src names a logo, and does not
/// name a customer grid. Two or more such images are ambiguous and ignored.
fn find_single_unambiguous_logo(doc: &Html, base_url: Option<&Url>) -> Option<String> {
    let mut found: Option<String> = None;
    for el in doc.select(selector!("img")) {
        if image_has_excluded_logo_term(el) {
            continue;
        }
        let class = el.value().attr("class").unwrap_or("");
        let id = el.value().attr("id").unwrap_or("");
        let alt = el.value().attr("alt").unwrap_or("");
        let src = el.value().attr("src").unwrap_or("").trim();
        let signal = [class, id, alt, src]
            .iter()
            .any(|field| contains_ci(field, "logo"));
        if !signal || src.is_empty() {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(resolve_url(src, base_url));
    }
    found
}

fn image_has_excluded_logo_term(el: ElementRef<'_>) -> bool {
    let srcset_attr = el.value().attr("srcset").unwrap_or("");
    let srcset = first_srcset_candidate(srcset_attr);
    [
        el.value().attr("class").unwrap_or(""),
        el.value().attr("id").unwrap_or(""),
        el.value().attr("alt").unwrap_or(""),
        el.value().attr("aria-label").unwrap_or(""),
        el.value().attr("src").unwrap_or(""),
        srcset,
    ]
    .iter()
    .copied()
    .any(contains_excluded_logo_term)
}

fn contains_excluded_logo_term(field: &str) -> bool {
    LOGO_EXCLUDE_TERMS
        .iter()
        .any(|term| contains_ci(field, term))
}

fn first_srcset_candidate(srcset: &str) -> &str {
    let first = srcset.split(',').next().unwrap_or("").trim();
    first.split_whitespace().next().unwrap_or("")
}

// ---------------------------------------------------------------------------
// Favicon / icon-link detection
// ---------------------------------------------------------------------------

/// Best `<link rel>` whose rel contains "icon". Raster icons outrank an SVG
/// that happens to be listed first. The returned score is 1 for SVG and
/// mask-icon (callers that need a servable bitmap skip those).
fn find_best_icon(doc: &Html, base_url: Option<&Url>) -> Option<(String, u32)> {
    let mut best: Option<(String, u32)> = None;
    for el in doc.select(selector!("link[rel]")) {
        let rel = el.value().attr("rel").unwrap_or("");
        if !contains_ci(rel, "icon") {
            continue;
        }
        let Some(href) = el.value().attr("href") else {
            continue;
        };
        let href = href.trim();
        if href.is_empty() {
            continue;
        }
        let sizes = el.value().attr("sizes").unwrap_or("");
        let score = icon_link_score(rel, href, sizes);
        let replace = best
            .as_ref()
            .is_none_or(|(_, best_score)| score > *best_score);
        if replace {
            best = Some((resolve_url(href, base_url), score));
        }
    }
    best
}

fn find_favicon(doc: &Html, base_url: Option<&Url>) -> Option<String> {
    find_best_icon(doc, base_url).map(|(url, _score)| url)
}

/// Higher wins. SVG and `mask-icon` stay at 1 so a raster icon is preferred
/// and so they are not promoted to `logo_url`.
fn icon_link_score(rel: &str, href: &str, sizes: &str) -> u32 {
    let ext = image_extension(href).unwrap_or_default();
    if contains_ci(rel, "mask-icon") || ext == "svg" {
        return 1;
    }
    let (rank, default_size) = match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "avif" => (3_u32, 48_u32),
        "ico" => (2, 32),
        _ => (2, 48),
    };
    let mut size = icon_pixel_size(sizes, href);
    if size == 0 {
        size = if contains_ci(rel, "apple") {
            180
        } else {
            default_size
        };
    }
    let apple = if contains_ci(rel, "apple") { 80 } else { 0 };
    rank * 1000 + apple + size.min(1024)
}

fn icon_pixel_size(sizes: &str, url: &str) -> u32 {
    let mut best = 0_u32;
    for text in [sizes, url] {
        for cap in ICON_DIMENSION.captures_iter(text) {
            let width: u32 = cap[1].parse().unwrap_or(0);
            let height: u32 = cap[2].parse().unwrap_or(0);
            best = best.max(width.max(height));
        }
    }
    if best > 0 {
        return best;
    }
    let from_query = query_pixel_size(url);
    if from_query > 0 {
        return from_query;
    }
    filename_token_size(url)
}

fn query_pixel_size(url: &str) -> u32 {
    let Some((_, query)) = url.split_once('?') else {
        return 0;
    };
    let query = query.split('#').next().unwrap_or(query);
    let mut best = 0_u32;
    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        let key = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("");
        if matches!(
            key.to_ascii_lowercase().as_str(),
            "w" | "h" | "width" | "height"
        ) && let Ok(pixels) = value.parse::<u32>()
        {
            best = best.max(pixels);
        }
    }
    best
}

fn filename_token_size(url: &str) -> u32 {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let file = path.rsplit('/').next().unwrap_or(path);
    let mut best = 0_u32;
    for cap in ICON_FILENAME_SIZE.captures_iter(file) {
        if let Ok(pixels) = cap[1].parse::<u32>() {
            best = best.max(pixels);
        }
    }
    best
}

fn image_extension(url: &str) -> Option<String> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let (_, ext) = file.rsplit_once('.')?;
    if ext.is_empty() || ext.len() > 8 || !ext.chars().all(|ch| ch.is_ascii_alphanumeric()) {
        return None;
    }
    Some(ext.to_ascii_lowercase())
}

fn is_raster_image_url(url: &str) -> bool {
    matches!(
        image_extension(url).as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "gif" | "avif" | "ico")
    )
}

/// Raster `Organization.logo` from JSON-LD, including `@graph`. SVG wordmarks
/// are ignored so an icon or public favicon can still be served.
fn find_jsonld_logo(doc: &Html, base_url: Option<&Url>) -> Option<String> {
    for el in doc.select(selector!("script")) {
        let kind = el.value().attr("type").unwrap_or("");
        if !contains_ci(kind, "ld+json") {
            continue;
        }
        let text: String = el.text().collect();
        let Ok(value) = serde_json::from_str::<serde_json::Value>(text.trim()) else {
            continue;
        };
        if let Some(raw) = organization_raster_logo(&value) {
            return Some(resolve_url(&raw, base_url));
        }
    }
    None
}

fn organization_raster_logo(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Array(items) => items.iter().find_map(organization_raster_logo),
        serde_json::Value::Object(map) => {
            if map.get("@type").is_some_and(is_organization_type)
                && let Some(logo) = map.get("logo").and_then(raster_logo_value)
            {
                return Some(logo);
            }
            for (key, child) in map {
                if key == "@context" || key == "@type" || key == "logo" {
                    continue;
                }
                if let Some(logo) = organization_raster_logo(child) {
                    return Some(logo);
                }
            }
            None
        }
        _ => None,
    }
}

fn is_organization_type(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(raw) => schema_type_is_organization(raw),
        serde_json::Value::Array(items) => items.iter().any(is_organization_type),
        _ => false,
    }
}

fn schema_type_is_organization(raw: &str) -> bool {
    let name = raw.trim().rsplit(['/', ':']).next().unwrap_or(raw).trim();
    name.eq_ignore_ascii_case("organization")
}

fn raster_logo_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(url) => raster_logo_string(url),
        serde_json::Value::Object(map) => map
            .get("url")
            .and_then(|url| url.as_str())
            .and_then(raster_logo_string),
        serde_json::Value::Array(items) => items.iter().find_map(raster_logo_value),
        _ => None,
    }
}

fn raster_logo_string(url: &str) -> Option<String> {
    let url = url.trim();
    if url.is_empty() || !is_raster_image_url(url) {
        return None;
    }
    Some(url.to_string())
}

/// `https://www.google.com/s2/favicons?domain={host}&sz=128`
fn public_favicon_url(page: &Url) -> Option<String> {
    let host = page.host_str()?;
    if host.is_empty() {
        return None;
    }
    let mut favicon = Url::parse("https://www.google.com/s2/favicons").ok()?;
    favicon
        .query_pairs_mut()
        .append_pair("domain", host)
        .append_pair("sz", "128");
    Some(favicon.to_string())
}

// ---------------------------------------------------------------------------
// Brand name extraction
// ---------------------------------------------------------------------------

/// Extract brand name from metadata, with fallback chain:
/// 1. og:site_name
/// 2. <meta name="application-name">
/// 3. <title> (cleaned up — strip " - Home", " | Official", etc.)
fn extract_brand_name(doc: &Html) -> Option<String> {
    // og:site_name
    for el in doc.select(selector!("meta[property='og:site_name']")) {
        if let Some(content) = el.value().attr("content") {
            let name = content.trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }

    // application-name
    for el in doc.select(selector!("meta[name='application-name']")) {
        if let Some(content) = el.value().attr("content") {
            let name = content.trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }

    // <title> with cleanup
    for el in doc.select(selector!("title")) {
        let title: String = el.text().collect();
        let title = title.trim();
        if !title.is_empty() {
            return Some(clean_title_to_brand(title));
        }
    }

    None
}

/// Clean a page <title> to extract just the brand name.
/// "Akari Corporation - Home" → "Akari Corporation"
/// "Products | Vercel" → "Vercel" (take the shorter part)
fn clean_title_to_brand(title: &str) -> String {
    // Split on common separators
    for sep in [" | ", " - ", " — ", " · ", " :: ", " // "] {
        if let Some(pos) = title.find(sep) {
            let left = title[..pos].trim();
            let right = title[pos + sep.len()..].trim();
            // Common suffixes that indicate the left part is the page name, not the brand
            let page_suffixes = ["home", "homepage", "official", "welcome"];
            if page_suffixes
                .iter()
                .any(|s| left.to_lowercase().ends_with(s))
            {
                return left.to_string();
            }
            if page_suffixes
                .iter()
                .any(|s| right.to_lowercase().ends_with(s))
            {
                return left.to_string();
            }
            // Take the shorter part (brand names tend to be shorter than page titles)
            if right.len() < left.len() && right.len() >= 2 {
                return right.to_string();
            }
            return left.to_string();
        }
    }
    title.to_string()
}

// ---------------------------------------------------------------------------
// Multi-logo detection
// ---------------------------------------------------------------------------

/// Find ALL logo variants from the page.
fn find_all_logos(doc: &Html, base_url: Option<&Url>) -> Vec<LogoVariant> {
    let mut logos = Vec::new();
    let mut seen_urls: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut add = |url: String, kind: &str| {
        if !url.is_empty() && seen_urls.insert(url.clone()) {
            logos.push(LogoVariant {
                url,
                kind: kind.to_string(),
            });
        }
    };

    // Favicons (rel="icon", rel="shortcut icon")
    for el in doc.select(selector!("link[rel]")) {
        let rel = el.value().attr("rel").unwrap_or("").to_lowercase();
        if let Some(href) = el.value().attr("href")
            && rel.contains("icon")
            && !rel.contains("apple")
        {
            add(resolve_url(href, base_url), "favicon");
        }
    }

    // Apple touch icons
    for el in doc.select(selector!("link[rel='apple-touch-icon']")) {
        if let Some(href) = el.value().attr("href") {
            add(resolve_url(href, base_url), "apple-touch-icon");
        }
    }
    for el in doc.select(selector!("link[rel='apple-touch-icon-precomposed']")) {
        if let Some(href) = el.value().attr("href") {
            add(resolve_url(href, base_url), "apple-touch-icon");
        }
    }

    // Logo images in header/nav first. Product/customer logo grids elsewhere
    // are common on SaaS sites and should not become the primary brand signal.
    for el in doc.select(selector!("header img, nav img")) {
        let class = el.value().attr("class").unwrap_or("");
        let id = el.value().attr("id").unwrap_or("");
        let alt = el.value().attr("alt").unwrap_or("");
        if (contains_ci(class, "logo") || contains_ci(id, "logo") || contains_ci(alt, "logo"))
            && let Some(src) = el.value().attr("src")
        {
            add(resolve_url(src, base_url), "logo");
        }
    }

    // Inline SVGs that look like logos (in header/nav, with viewBox)
    for el in doc.select(selector!(
        "header svg[viewBox], nav svg[viewBox], a svg[viewBox]"
    )) {
        // We can't return SVG content as a URL, but we can note it exists
        if logos.iter().all(|l| l.kind != "svg") {
            // Try to find if the SVG is wrapped in a link to homepage
            if let Some(parent) = el.parent().and_then(|p| p.value().as_element())
                && parent
                    .attr("href")
                    .is_some_and(|h| h == "/" || h.ends_with(".com") || h.ends_with(".com/"))
            {
                logos.push(LogoVariant {
                    url: "(inline-svg)".to_string(),
                    kind: "svg".to_string(),
                });
            }
        }
    }

    logos
}

/// Find Open Graph image.
fn find_og_image(doc: &Html, base_url: Option<&Url>) -> Option<String> {
    for el in doc.select(selector!("meta[property='og:image']")) {
        if let Some(content) = el.value().attr("content") {
            let url = content.trim();
            if !url.is_empty() {
                return Some(resolve_url(url, base_url));
            }
        }
    }
    // Twitter card image as fallback
    for el in doc.select(selector!("meta[name='twitter:image']")) {
        if let Some(content) = el.value().attr("content") {
            let url = content.trim();
            if !url.is_empty() {
                return Some(resolve_url(url, base_url));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contains_ci(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

fn resolve_url(src: &str, base_url: Option<&Url>) -> String {
    match base_url {
        Some(base) => base
            .join(src)
            .map(|u| u.to_string())
            .unwrap_or_else(|_| src.to_string()),
        None => src.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_colors() {
        let html = r#"<html><head><style>
            .header { background-color: #3498db; }
            .text { color: #2c3e50; }
            .border { border-color: #e74c3c; }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        assert!(!brand.colors.is_empty(), "should extract colors");

        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#3498DB"), "should find header bg color");
        assert!(hexes.contains(&"#2C3E50"), "should find text color");
        assert!(hexes.contains(&"#E74C3C"), "should find border color");

        // Verify usage classification
        let bg = brand.colors.iter().find(|c| c.hex == "#3498DB").unwrap();
        assert_eq!(bg.usage, ColorUsage::Background);

        let text = brand.colors.iter().find(|c| c.hex == "#2C3E50").unwrap();
        assert_eq!(text.usage, ColorUsage::Text);

        let accent = brand.colors.iter().find(|c| c.hex == "#E74C3C").unwrap();
        assert_eq!(accent.usage, ColorUsage::Accent);
    }

    #[test]
    fn test_short_hex_expansion() {
        let html = r#"<html><head><style>
            .x { color: #f00; }
            .y { background-color: #0af; }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#FF0000"), "#f00 should expand to #FF0000");
        assert!(hexes.contains(&"#00AAFF"), "#0af should expand to #00AAFF");
    }

    #[test]
    fn test_rgb_colors() {
        let html = r#"<html><head><style>
            .btn { background-color: rgb(52, 152, 219); }
            .link { color: rgba(231, 76, 60, 0.8); }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#3498DB"), "rgb(52,152,219) -> #3498DB");
        assert!(hexes.contains(&"#E74C3C"), "rgba(231,76,60,0.8) -> #E74C3C");
    }

    #[test]
    fn test_hsl_colors() {
        let html = r#"<html><head><style>
            .x { color: hsl(0, 100%, 50%); }
            .y { background-color: hsla(240, 100%, 50%, 0.5); }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#FF0000"), "hsl(0,100%,50%) -> #FF0000");
        assert!(hexes.contains(&"#0000FF"), "hsla(240,100%,50%) -> #0000FF");
    }

    #[test]
    fn test_boring_colors_filtered() {
        let html = r#"<html><head><style>
            body { background-color: #ffffff; color: #000000; }
            .gray { color: #cccccc; }
            .brand { color: #3498db; }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(!hexes.contains(&"#FFFFFF"), "white should be filtered");
        assert!(!hexes.contains(&"#000000"), "black should be filtered");
        assert!(
            !hexes.contains(&"#CCCCCC"),
            "common gray should be filtered"
        );
        assert!(hexes.contains(&"#3498DB"), "brand color should survive");
    }

    #[test]
    fn test_google_oauth_palette_does_not_overwhelm_non_google_brand() {
        let html = r#"<html><head>
            <meta property="og:site_name" content="Claude">
            <style>
                .google-button { color: #1A73E8; background: #4285F4; border-color: #5F6368; }
                .google-icon { color: #202124; background: #E8EAED; }
                :root { --brand-accent: #D97757; --brand-text: #DC6038; }
            </style>
        </head><body></body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(!hexes.contains(&"#1A73E8"));
        assert!(!hexes.contains(&"#4285F4"));
        assert!(hexes.contains(&"#D97757"));
        assert!(hexes.contains(&"#DC6038"));
    }

    #[test]
    fn test_font_extraction() {
        let html = r#"<html><head><style>
            body { font-family: "Inter", "Helvetica Neue", sans-serif; }
            code { font-family: 'Fira Code', monospace; }
            h1 { font-family: Inter, sans-serif; }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        assert!(
            brand.fonts.contains(&"Inter".to_string()),
            "should find Inter"
        );
        assert!(
            brand.fonts.contains(&"Helvetica Neue".to_string()),
            "should find Helvetica Neue"
        );
        assert!(
            brand.fonts.contains(&"Fira Code".to_string()),
            "should find Fira Code"
        );
        // Generic families should be excluded
        assert!(!brand.fonts.contains(&"sans-serif".to_string()));
        assert!(!brand.fonts.contains(&"monospace".to_string()));
    }

    #[test]
    fn test_font_ordering_by_frequency() {
        let html = r#"<html><head><style>
            body { font-family: "Inter", sans-serif; }
            p { font-family: "Inter", sans-serif; }
            h1 { font-family: "Inter", sans-serif; }
            code { font-family: "Fira Code", monospace; }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        assert!(!brand.fonts.is_empty());
        assert_eq!(
            brand.fonts[0], "Inter",
            "most frequent font should be first"
        );
    }

    #[test]
    fn test_font_shorthand_is_normalized_and_noise_filtered() {
        let html = r#"<html><head><style>
            body { font: 500 12px "Roboto", Arial, sans-serif; }
            h1 { font: 1.21em/1.2 KaTeX_Main; }
            .hash { font-family: "9d9927955a95a20d s", "OpenAI Sans", sans-serif; }
            .bad { font-family: "Noto Color Emoji\")", "Segoe UI Emoji"; }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        assert!(brand.fonts.contains(&"Roboto".to_string()));
        assert!(brand.fonts.contains(&"OpenAI Sans".to_string()));
        assert!(!brand.fonts.iter().any(|f| f.contains("12px")));
        assert!(!brand.fonts.iter().any(|f| f.contains("KaTeX")));
        assert!(!brand.fonts.iter().any(|f| f.contains("Emoji")));
        assert!(!brand.fonts.iter().any(|f| f.contains("9d9927955a95a20d")));
    }

    #[test]
    fn test_logo_by_class() {
        let html = r#"<html><body>
            <header>
                <img class="site-logo" src="/images/logo.png" alt="Company">
                <img src="/images/banner.jpg" alt="Banner">
            </header>
        </body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/images/logo.png")
        );
    }

    #[test]
    fn test_logo_by_id() {
        let html = r#"<html><body>
            <header>
                <img id="main-logo" src="/logo.svg" alt="Brand">
            </header>
        </body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/logo.svg")
        );
    }

    #[test]
    fn test_logo_by_alt() {
        let html = r#"<html><body>
            <header>
                <img src="/brand-logo.png" alt="Acme Corp Logo">
            </header>
        </body></html>"#;

        let brand = extract_brand(html, Some("https://acme.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://acme.com/brand-logo.png")
        );
    }

    #[test]
    fn test_body_logo_grid_does_not_become_primary_brand_logo() {
        let html = r#"<html><body>
            <main>
                <section class="customers">
                    <img class="customer-logo" src="/logos/runway.svg" alt="Runway logo">
                    <img class="customer-logo" src="/logos/zapier.svg" alt="Zapier logo">
                </section>
            </main>
        </body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        let logo = brand
            .logo_url
            .expect("host should fall back to a public favicon");
        assert!(
            logo.contains("domain=example.com"),
            "customer grid must not be selected, got {logo}"
        );
        assert!(logo.contains("sz=128"), "{logo}");
        assert!(
            !logo.contains("runway") && !logo.contains("zapier"),
            "{logo}"
        );
        assert_eq!(brand.favicon_url.as_deref(), Some(logo.as_str()));
        assert!(
            brand
                .logos
                .iter()
                .all(|entry| { !entry.url.contains("runway") && !entry.url.contains("zapier") })
        );
        assert!(
            brand
                .logos
                .iter()
                .any(|entry| entry.kind == "favicon" && entry.url == logo)
        );
    }

    #[test]
    fn test_header_logo_is_still_primary_logo() {
        let html = r#"<html><body>
            <header>
                <img class="brand-logo" src="/logo.svg" alt="Acme logo">
            </header>
            <main>
                <img class="customer-logo" src="/logos/customer.svg" alt="Customer logo">
            </main>
        </body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/logo.svg")
        );
        assert_eq!(brand.logos.len(), 1);
        assert_eq!(brand.logos[0].url, "https://example.com/logo.svg");
    }

    #[test]
    fn test_favicon() {
        let html = r#"<html><head>
            <link rel="icon" href="/favicon.ico">
        </head><body></body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.favicon_url.as_deref(),
            Some("https://example.com/favicon.ico")
        );
    }

    #[test]
    fn test_favicon_shortcut_icon() {
        let html = r#"<html><head>
            <link rel="shortcut icon" href="/img/fav.png">
        </head><body></body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.favicon_url.as_deref(),
            Some("https://example.com/img/fav.png")
        );
    }

    #[test]
    fn test_full_brand() {
        let html = r#"
        <html>
        <head>
            <link rel="icon" href="/favicon.ico">
            <style>
                body {
                    font-family: "Roboto", "Open Sans", sans-serif;
                    background-color: #f5f5f5;
                    color: #2d3436;
                }
                .header { background-color: #6c5ce7; }
                .btn-primary { background-color: #6c5ce7; color: #ffeaa7; }
                .btn-secondary { background-color: #00b894; }
                a { color: #0984e3; }
                .border { border-color: #dfe6e9; }
                code { font-family: "JetBrains Mono", monospace; }
            </style>
        </head>
        <body>
            <header class="header">
                <a href="/"><img class="logo" src="/images/logo.svg" alt="Brand"></a>
                <nav>
                    <a href="/about">About</a>
                </nav>
            </header>
            <main>
                <h1>Welcome</h1>
                <p>Hello world</p>
            </main>
        </body>
        </html>"#;

        let brand = extract_brand(html, Some("https://example.com"));

        // Colors
        assert!(!brand.colors.is_empty(), "should extract colors");
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#6C5CE7"), "should find primary purple");
        assert!(hexes.contains(&"#0984E3"), "should find link blue");
        assert!(hexes.contains(&"#00B894"), "should find secondary green");
        // #f5f5f5 is a boring gray, should be filtered
        assert!(
            !hexes.contains(&"#F5F5F5"),
            "boring gray should be filtered"
        );

        // Fonts
        assert!(brand.fonts.contains(&"Roboto".to_string()));
        assert!(brand.fonts.contains(&"Open Sans".to_string()));
        assert!(brand.fonts.contains(&"JetBrains Mono".to_string()));
        assert!(!brand.fonts.contains(&"sans-serif".to_string()));
        assert!(!brand.fonts.contains(&"monospace".to_string()));

        // Logo
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/images/logo.svg")
        );

        // Favicon
        assert_eq!(
            brand.favicon_url.as_deref(),
            Some("https://example.com/favicon.ico")
        );
    }

    #[test]
    fn test_inline_styles() {
        let html = r#"<html><body>
            <div style="background-color: #e74c3c; color: #ecf0f1;">Content</div>
            <span style="font-family: 'Poppins', sans-serif;">Text</span>
        </body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#E74C3C"), "should find inline bg color");
        assert!(hexes.contains(&"#ECF0F1"), "should find inline text color");
        assert!(brand.fonts.contains(&"Poppins".to_string()));
    }

    #[test]
    fn test_no_logo_or_favicon() {
        let html = r#"<html><head></head><body><p>Simple page</p></body></html>"#;
        let brand = extract_brand(html, None);
        assert!(brand.logo_url.is_none());
        assert!(brand.favicon_url.is_none());
    }

    #[test]
    fn test_empty_html() {
        let brand = extract_brand("", None);
        assert!(brand.colors.is_empty());
        assert!(brand.fonts.is_empty());
        assert!(brand.logo_url.is_none());
        assert!(brand.favicon_url.is_none());
    }

    #[test]
    fn test_css_custom_properties() {
        let html = r#"<html><head><style>
            :root {
                --primary: #3b82f6;
                --bg-dark: rgb(15, 23, 42);
                --accent: hsl(340, 82%, 52%);
                --spacing: 1rem; /* not a color */
            }
        </style></head><body></body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#3B82F6"), "should find --primary");
        assert!(hexes.contains(&"#0F172A"), "should find --bg-dark");
        assert!(
            brand.colors.len() >= 3,
            "should find at least 3 colors from vars"
        );
    }

    #[test]
    fn test_tailwind_arbitrary_colors() {
        let html = r#"<html><body>
            <div class="bg-[#1a1a2e] text-[#e94560]">Content</div>
            <span class="border-[rgb(255,107,107)]">Border</span>
        </body></html>"#;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#1A1A2E"), "should find bg-[#1a1a2e]");
        assert!(hexes.contains(&"#E94560"), "should find text-[#e94560]");
        assert!(
            hexes.contains(&"#FF6B6B"),
            "should find border-[rgb(255,107,107)]"
        );
    }

    #[test]
    fn test_theme_color_meta() {
        let html = r##"<html><head>
            <meta name="theme-color" content="#6366f1">
        </head><body></body></html>"##;

        let brand = extract_brand(html, None);
        let hexes: Vec<&str> = brand.colors.iter().map(|c| c.hex.as_str()).collect();
        assert!(hexes.contains(&"#6366F1"), "should find theme-color");
    }

    #[test]
    fn test_google_fonts_url() {
        let html = r#"<html><head>
            <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Roboto+Mono:wght@400">
        </head><body></body></html>"#;

        let brand = extract_brand(html, None);
        assert!(
            brand.fonts.contains(&"Inter".to_string()),
            "should find Inter from Google Fonts URL"
        );
        assert!(
            brand.fonts.contains(&"Roboto Mono".to_string()),
            "should find Roboto Mono from Google Fonts URL"
        );
    }

    #[test]
    fn test_font_preload() {
        let html = r#"<html><head>
            <link rel="preload" as="font" href="/fonts/Geist-Variable.woff2" crossorigin>
            <link rel="preload" as="font" href="/fonts/GeistMono-Regular.woff2" crossorigin>
        </head><body></body></html>"#;

        let brand = extract_brand(html, None);
        assert!(
            brand.fonts.iter().any(|f| f.contains("Geist")),
            "should find Geist from preload"
        );
    }

    #[test]
    fn test_extract_font_name_from_url() {
        assert_eq!(
            extract_font_name_from_url("/fonts/Inter-Variable.woff2"),
            Some("Inter".to_string())
        );
        assert_eq!(
            extract_font_name_from_url("/fonts/Geist-Regular.woff2"),
            Some("Geist".to_string())
        );
        assert_eq!(
            extract_font_name_from_url("/fonts/JetBrainsMono-Bold.woff2"),
            Some("JetBrainsMono".to_string())
        );
    }

    #[test]
    fn test_google_fonts_from_url() {
        let fonts = extract_google_fonts_from_url(
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Roboto+Mono:wght@400",
        );
        assert!(fonts.contains(&"Inter".to_string()));
        assert!(fonts.contains(&"Roboto Mono".to_string()));
    }

    #[test]
    fn test_max_10_colors() {
        // Generate HTML with 15 distinct colors
        let colors: Vec<String> = (0..15)
            .map(|i| {
                format!(
                    ".c{i} {{ color: #{:02X}{:02X}{:02X}; }}",
                    10 + i * 15,
                    20 + i * 10,
                    30 + i * 5
                )
            })
            .collect();
        let html = format!(
            "<html><head><style>{}</style></head><body></body></html>",
            colors.join("\n")
        );

        let brand = extract_brand(&html, None);
        assert!(brand.colors.len() <= 10, "should cap at 10 colors");
    }

    #[test]
    fn test_favicon_prefers_png_over_svg() {
        let html = r#"<html><head>
            <link rel="icon" href="/favicon.svg" type="image/svg+xml">
            <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
        </head><body></body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.favicon_url.as_deref(),
            Some("https://example.com/favicon-32.png")
        );
    }

    #[test]
    fn test_icon_only_page_promotes_apple_touch_png() {
        // Stripe-shaped: inline header SVG is not a URL, and the 180px
        // apple-touch PNG outranks a 96px PNG and an SVG icon.
        let html = r#"<html><head>
            <link rel="icon" href="/favicon.svg" type="image/svg+xml">
            <link rel="icon" type="image/png" sizes="96x96" href="/favicon-96x96.png">
            <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
        </head><body>
            <header>
                <a href="/" aria-label="Stripe">
                    <svg viewBox="0 0 24 24"><path d="M0 0h24v24H0z"/></svg>
                </a>
            </header>
        </body></html>"#;

        let brand = extract_brand(html, Some("https://stripe.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://stripe.com/apple-touch-icon.png")
        );
    }

    #[test]
    fn test_jsonld_raster_logo_beats_favicon() {
        let string_logo = r#"<html><head>
            <link rel="icon" href="/favicon.ico">
            <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": "Organization",
              "name": "Example",
              "logo": "https://cdn.example.com/logo.png"
            }
            </script>
        </head><body></body></html>"#;
        let brand = extract_brand(string_logo, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://cdn.example.com/logo.png")
        );
        assert_eq!(
            brand.favicon_url.as_deref(),
            Some("https://example.com/favicon.ico")
        );

        let graph_logo = r#"<html><head>
            <link rel="icon" href="/favicon.ico">
            <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@graph": [
                {
                  "@type": "Organization",
                  "logo": {
                    "@type": "ImageObject",
                    "url": "https://cdn.example.com/brand.jpg"
                  }
                }
              ]
            }
            </script>
        </head><body></body></html>"#;
        let brand = extract_brand(graph_logo, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://cdn.example.com/brand.jpg")
        );

        let svg_logo = r#"<html><head>
            <link rel="icon" href="/favicon.ico">
            <script type="application/ld+json">
            {"@type":"Organization","logo":"https://cdn.example.com/logo.svg"}
            </script>
        </head><body></body></html>"#;
        let brand = extract_brand(svg_logo, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/favicon.ico"),
            "non-raster JSON-LD logos are not servable wordmarks"
        );
    }

    #[test]
    fn test_header_svg_logo_beats_apple_touch_icon() {
        let html = r#"<html><head>
            <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
            <link rel="icon" href="/favicon.ico">
        </head><body>
            <header><img class="logo" src="/logo.svg" alt="Acme"></header>
        </body></html>"#;

        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/logo.svg")
        );
    }

    #[test]
    fn test_challenge_html_uses_public_favicon() {
        let html = r#"<html><head><title>Just a moment...</title></head>
            <body><p>Checking your browser</p></body></html>"#;
        let brand = extract_brand(html, Some("https://openai.com/"));
        let logo = brand.logo_url.expect("empty document with a host");
        assert!(logo.contains("domain=openai.com"), "{logo}");
        assert!(logo.contains("sz=128"), "{logo}");
        assert!(logo.starts_with("https://www.google.com/s2/favicons?"));
        assert_eq!(brand.favicon_url.as_deref(), Some(logo.as_str()));
        assert!(
            brand
                .logos
                .iter()
                .any(|entry| entry.kind == "favicon" && entry.url == logo)
        );
    }

    #[test]
    fn test_srcset_header_logo() {
        let html = r#"<html><body>
            <header>
                <img srcset="/images/logo-wordmark.png 1x, /images/logo-wordmark@2x.png 2x" alt="Acme">
            </header>
        </body></html>"#;
        let brand = extract_brand(html, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/images/logo-wordmark.png")
        );

        let src_signal = r#"<html><body>
            <header><img src="/assets/logo.png" alt="Acme"></header>
        </body></html>"#;
        let brand = extract_brand(src_signal, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/assets/logo.png")
        );

        let aria = r#"<html><body>
            <header>
                <img aria-label="Acme logo" srcset="https://cdn.example.com/mark.png 200w">
            </header>
        </body></html>"#;
        let brand = extract_brand(aria, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://cdn.example.com/mark.png")
        );
    }

    #[test]
    fn test_single_body_logo_is_used_until_a_second_appears() {
        let one = r#"<html><body>
            <img class="brand-logo" src="/only-logo.png" alt="Acme">
        </body></html>"#;
        let brand = extract_brand(one, Some("https://example.com"));
        assert_eq!(
            brand.logo_url.as_deref(),
            Some("https://example.com/only-logo.png")
        );

        let two = r#"<html><body>
            <img class="brand-logo" src="/a-logo.png" alt="A logo">
            <img id="site-logo" src="/b-logo.png" alt="B logo">
        </body></html>"#;
        let brand = extract_brand(two, Some("https://example.com"));
        let logo = brand.logo_url.expect("ambiguous body logos fall back");
        assert!(logo.contains("domain=example.com"), "{logo}");
        assert!(
            !logo.contains("a-logo") && !logo.contains("b-logo"),
            "{logo}"
        );
    }

    #[test]
    fn test_icon_link_score_prefers_raster_and_reads_size() {
        assert_eq!(icon_link_score("icon", "/favicon.svg", ""), 1);
        assert_eq!(icon_link_score("mask-icon", "/pinned.png", "16x16"), 1);
        assert_eq!(
            icon_link_score("apple-touch-icon", "/logo.svg", "180x180"),
            1
        );
        assert_eq!(icon_link_score("icon", "/favicon.ico", ""), 2_000 + 32);
        assert_eq!(
            icon_link_score("icon", "/favicon-32.png", "32x32"),
            3_000 + 32
        );
        assert_eq!(
            icon_link_score("icon", "/favicon-32x32.png?w=256", ""),
            3_000 + 32,
            "NxN in the URL beats a larger query size"
        );
        assert_eq!(
            icon_link_score("icon", "/logo.png?w=128&h=64", ""),
            3_000 + 128
        );
        assert_eq!(icon_link_score("icon", "/touch-76.png", ""), 3_000 + 76);
        assert_eq!(icon_link_score("icon", "/icon-256.png", ""), 3_000 + 256);
        assert_eq!(icon_link_score("icon", "/favicon-32.png", ""), 3_000 + 32);
        assert_eq!(
            icon_link_score("apple-touch-icon", "/apple-touch-icon.png", "any"),
            3_000 + 80 + 180
        );
        assert_eq!(
            icon_link_score("apple-touch-icon", "/apple-touch-icon.png", "180x180"),
            3_000 + 80 + 180
        );
        assert_eq!(
            icon_link_score("icon", "/favicon-96x96.png", ""),
            3_000 + 96
        );
        assert!(
            icon_link_score("apple-touch-icon", "/apple-touch-icon.png", "180x180")
                > icon_link_score("icon", "/favicon-96x96.png", "96x96")
        );
    }

    #[test]
    fn test_popular_domains_get_servable_logo_url() {
        let icon_only = r#"<html><head>
            <link rel="icon" href="/favicon.svg" type="image/svg+xml">
            <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
            <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
        </head><body>
            <header><a href="/"><svg viewBox="0 0 32 32" aria-label="Logo"></svg></a></header>
        </body></html>"#;
        let empty = r#"<html><head><title>Just a moment...</title></head>
            <body><p>Enable JavaScript and cookies to continue</p></body></html>"#;
        let wordmark = |attrs: &str, src: &str| {
            format!(
                r#"<html><head>
                    <link rel="icon" href="/favicon.svg" type="image/svg+xml">
                    <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
                </head><body>
                    <header><img {attrs} src="{src}" alt="Brand"></header>
                </body></html>"#
            )
        };

        let icon_domains = [
            "stripe.com",
            "notion.so",
            "linear.app",
            "github.com",
            "vercel.com",
            "figma.com",
            "spotify.com",
            "netflix.com",
            "airbnb.com",
            "cloudflare.com",
            "anthropic.com",
        ];
        let empty_domains = [
            "openai.com",
            "apple.com",
            "amazon.com",
            "nvidia.com",
            "atlassian.com",
        ];
        let wordmarks = [
            (
                "slack.com",
                wordmark("class=\"logo\"", "/img/slack-logo.png"),
                "/img/slack-logo.png",
            ),
            (
                "google.com",
                wordmark(
                    "id=\"hplogo\"",
                    "/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
                ),
                "/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
            ),
            (
                "discord.com",
                wordmark("class=\"logo\"", "/assets/discord-logo.svg"),
                "/assets/discord-logo.svg",
            ),
            (
                "shopify.com",
                wordmark("class=\"logo\"", "/brand/shopify-logo.svg"),
                "/brand/shopify-logo.svg",
            ),
            (
                "dropbox.com",
                wordmark("class=\"logo\"", "/brand/dropbox-logo.svg"),
                "/brand/dropbox-logo.svg",
            ),
            (
                "microsoft.com",
                wordmark("class=\"logo\"", "/images/microsoft-logo.jpg"),
                "/images/microsoft-logo.jpg",
            ),
        ];

        let mut cases: Vec<(&str, String, String)> = Vec::new();
        for domain in icon_domains {
            cases.push((
                domain,
                icon_only.to_string(),
                format!("https://{domain}/apple-touch-icon.png"),
            ));
        }
        for (domain, html, path) in wordmarks {
            cases.push((domain, html, format!("https://{domain}{path}")));
        }
        for domain in empty_domains {
            cases.push((
                domain,
                empty.to_string(),
                format!("https://www.google.com/s2/favicons?domain={domain}&sz=128"),
            ));
        }
        assert_eq!(cases.len(), 22);

        let mut new_hits = 0;
        let mut old_hits = 0;
        for (domain, html, expected) in &cases {
            let page = format!("https://{domain}/");
            let brand = extract_brand(html, Some(&page));
            let logo = brand.logo_url.unwrap_or_default();
            assert!(!logo.is_empty(), "{domain} produced an empty logo_url");
            assert_eq!(logo, *expected, "{domain}");
            new_hits += 1;
            if header_only_logo(html, &page).is_some() {
                old_hits += 1;
            }
        }
        assert_eq!(new_hits, cases.len(), "every fixture should hit");
        assert!(
            new_hits > old_hits,
            "servable fallback should beat the header-only finder ({new_hits} vs {old_hits})"
        );
    }

    /// Previous `find_logo`: header/nav class, id, or alt containing "logo",
    /// else an image inside a homepage link. No icon or host fallback.
    fn header_only_logo(html: &str, page_url: &str) -> Option<String> {
        let doc = Html::parse_document(html);
        let base = Url::parse(page_url).ok();
        let header = Selector::parse("header img, nav img").unwrap();
        for el in doc.select(&header) {
            let class = el.value().attr("class").unwrap_or("");
            let id = el.value().attr("id").unwrap_or("");
            let alt = el.value().attr("alt").unwrap_or("");
            if let Some(src) = el.value().attr("src")
                && (contains_ci(class, "logo")
                    || contains_ci(id, "logo")
                    || contains_ci(alt, "logo"))
            {
                return Some(resolve_url(src, base.as_ref()));
            }
        }
        let home = Selector::parse("a[href='/'] img, a[href] img").unwrap();
        for el in doc.select(&home) {
            if let Some(parent) = el.parent().and_then(|p| p.value().as_element()) {
                let href = parent.attr("href").unwrap_or("");
                if (href == "/" || href.ends_with(".com") || href.ends_with(".com/"))
                    && let Some(src) = el.value().attr("src")
                {
                    return Some(resolve_url(src, base.as_ref()));
                }
            }
        }
        None
    }
}
