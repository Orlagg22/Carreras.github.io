/**
 * The one source of the Noryx logo: the LC emblem (full and small), the outlined wordmark, the bars and the
 * lock-up proportions. `NoryxLogo.tsx` draws from it at runtime and `scripts/build-brand.ts` writes every file in
 * public/branding and public/favicon.ico from it (`npm run brand`), so no rendition is kept by hand.
 *
 * The hex values below are the mark's own colours and are defined only here. Chalk, volt, void, cyan, crimson and
 * slate are the same values as the `lc-*` tokens in globals.css; the badge, its edge and the two volt gradient stops
 * belong to the emblem alone.
 */

/** The one slant of the logo, in degrees: the emblem's glyphs, the wordmark and the three bars all lean this far */
export const LOGO_SLANT_DEG = 12;

export const LOGO_COLORS = {
  chalk: "#F5F7FA",
  volt: "#CCFF00",
  void: "#08090D",
  cyan: "#00E5FF",
  crimson: "#FF2A3C",
  slate: "#8E96AA",
  badgeTop: "#1A1E2B",
  badgeBottom: "#07080C",
  badgeEdge: "#2C3246",
  voltLight: "#E2FF4D",
  voltDeep: "#A6E000",
} as const;

/** The emblem is drawn in a 128 unit square; the badge fills 6 to 122 and its edge stroke reaches 5 to 123 */
export const EMBLEM_SIZE = 128;
export const EMBLEM_VIEWBOX = "0 0 128 128";
/** The badge and its edge with no margin: favicons and the downloadable emblem */
export const EMBLEM_TIGHT_VIEWBOX = "5 5 118 118";
/** Below this width in CSS px the small emblem takes over (no kerb, no streaks, heavier L and C) */
export const SMALL_EMBLEM_BELOW_PX = 32;

export type GradientId = "badge" | "volt" | "sheen";

export interface LogoGradient {
  id: GradientId;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  stops: readonly { offset: number; color: string; opacity?: number }[];
}

/** A colour, or one of the emblem's gradients */
export type Paint = string | { gradient: GradientId };

export interface LogoLayer {
  d: string;
  fill?: Paint;
  stroke?: Paint;
  strokeWidth?: number;
  strokeOpacity?: number;
  dash?: string;
  /** round line caps */
  round?: boolean;
}

/** Badge layers are drawn as they are; glyph layers inside `GLYPH_TRANSFORM` (the slant) */
export interface EmblemDrawing {
  badge: readonly LogoLayer[];
  glyphs: readonly LogoLayer[];
}

export const EMBLEM_GRADIENTS: readonly LogoGradient[] = [
  {
    id: "badge",
    x1: 12,
    y1: 6,
    x2: 116,
    y2: 122,
    stops: [
      { offset: 0, color: LOGO_COLORS.badgeTop },
      { offset: 1, color: LOGO_COLORS.badgeBottom },
    ],
  },
  {
    id: "volt",
    x1: 30,
    y1: 24,
    x2: 96,
    y2: 102,
    stops: [
      { offset: 0, color: LOGO_COLORS.voltLight },
      { offset: 1, color: LOGO_COLORS.voltDeep },
    ],
  },
  {
    id: "sheen",
    x1: 0,
    y1: 0,
    x2: 0,
    y2: 128,
    stops: [
      { offset: 0, color: "#FFFFFF", opacity: 0.16 },
      { offset: 0.5, color: "#FFFFFF", opacity: 0 },
    ],
  },
];

/** The glyphs lean by the logo's slant about the top edge; the shift keeps them centred on the badge */
export const GLYPH_TRANSFORM = `translate(13 0) skewX(-${LOGO_SLANT_DEG})`;

/** Badge: chamfered top-left and bottom-right, like a number board */
export const BADGE_PATH = "M28 6H110Q122 6 122 18V98L98 122H18Q6 122 6 110V28Z";

const L_PATH = "M32 100V26H48L82 74V26H96V100H80L46 52V100Z";
/** The C: centre (77, 57), radius 20, open to the right between 40 degrees above and below */
/** The kerb: the C's apex, the quarter facing the L's corner */

const badgeLayers = (edgeWidth: number, accent: LogoLayer): LogoLayer[] => [
  { d: BADGE_PATH, fill: { gradient: "badge" } },
  { d: BADGE_PATH, fill: { gradient: "sheen" } },
  { d: BADGE_PATH, stroke: LOGO_COLORS.badgeEdge, strokeWidth: edgeWidth },
  accent,
];

/**
 * The LC emblem: a forward-leaning volt L with a white C nested in its corner like a car taking an apex. The C's apex
 * is painted as a red and white kerb and a cyan streak leaves the exit.
 */
export const EMBLEM: EmblemDrawing = {
  badge: badgeLayers(2, { d: "M101 117.5L117.5 101", stroke: LOGO_COLORS.volt, strokeWidth: 4, round: true }),
  glyphs: [
    { d: L_PATH, fill: { gradient: "volt" } },
    { d: "M101 42H112", stroke: LOGO_COLORS.cyan, strokeWidth: 4.5, round: true },
    { d: "M104 51H110", stroke: LOGO_COLORS.cyan, strokeOpacity: 0.45, strokeWidth: 3, round: true },
  ],
};

/*
 * The small emblem, for anything under 32 px (favicons, HUD badges). Laid out on an 8 unit grid, one pixel at 16 px:
 * the L is 20 units thick, the C 14, and 8 units of badge keep them apart, so they stay two shapes in one colour.
 * No kerb and no streaks; the badge edge and the volt corner are heavier so they survive the downsample.
 */
const L_SMALL_PATH = "M30 102V24H50L80 68V24H100V102H80L50 58V102Z";
/** Centre (84, 48), radius 19, stroke 14: the hole is 24 units across, the outside 8 units clear of the L */

export const EMBLEM_SMALL: EmblemDrawing = {
  badge: badgeLayers(4, { d: "M99 113.9L113.9 99", stroke: LOGO_COLORS.volt, strokeWidth: 8, round: true }),
  glyphs: [
    { d: L_SMALL_PATH, fill: { gradient: "volt" } },
  ],
};

/**
 * Single-colour emblem: the badge in one colour with the glyphs cut out of it, so it reads on any background and as an
 * alpha-only (monochrome) app icon. The glyph layers are drawn, in order and inside `GLYPH_TRANSFORM`, into the badge's
 * mask: `cut` removes the badge there, `keep` puts it back. A ring of badge is kept round the C so the L and the C
 * never touch, and the kerb stays as stripes across the C's apex.
 */
export interface MonoLayer {
  d: string;
  op: "cut" | "keep";
  /** stroke width; without it the path is filled */
  strokeWidth?: number;
  dash?: string;
  round?: boolean;
}

export interface MonoEmblem {
  shape: string;
  glyphs: readonly MonoLayer[];
}


export const EMBLEM_MONO: MonoEmblem = {
  shape: BADGE_PATH,
  glyphs: [
    { d: L_PATH, op: "cut" },
    { d: "M101 42H112", op: "cut", strokeWidth: 4.5, round: true },
    { d: "M104 51H110", op: "cut", strokeWidth: 3, round: true },
  ],
};

export const EMBLEM_MONO_SMALL: MonoEmblem = {
  shape: BADGE_PATH,
  glyphs: [
    { d: L_SMALL_PATH, op: "cut" },
  ],
};

/** Mask colours: black removes, white keeps */
export const MASK_COLORS = { cut: "#000000", keep: "#FFFFFF" } as const;

/*
 * The wordmark, LIBRE + CORSA, outlined from Orbitron at weight 900 with the app's tracking and the logo's slant.
 * The box is one em tall, the baseline where a `line-height: 1` line box puts it, so it sits where the live text sat.
 * The slanted capitals reach `overhang` past `width` at the right; give the box overflow or padding for that.
 */
// <wordmark:generated> by scripts/brand/outline-text.py from Orbitron 900, do not edit by hand
export const WORDMARK = {
  /** One em, in font units; the box is one em tall with the baseline where `line-height: 1` puts it */
  em: 1000,
  /** Advance of the ten letters with their tracking (0.05 em after each, the last one too) */
  width: 4353,
  baseline: 884,
  capHeight: 720,
  /** How far the slanted capitals reach right of the advance box at cap height */
  overhang: 153,
  libre: "M56 884 209 164H369L670 644L772 164H929L776 884H616L313.5 402L211 884ZM1084 884Q1043.5 884 1014 864Q984.5 844 971.7 810.2Q958.8 776.5 967.5 736L1057.6 312Q1066.2 271.5 1093.4 237.8Q1120.5 204 1158.5 184Q1196.5 164 1237 164H1661Q1701.2 164 1730.4 184Q1759.5 204 1772.9 237.8Q1786.2 271.5 1777.6 312L1687.5 736Q1678.8 776.5 1651.2 810.2Q1623.5 844 1585.8 864Q1548.1 884 1508 884ZM1124.2 728H1532.2Q1532.2 728 1532.2 728Q1532.2 728 1532.2 728L1618.9 320Q1618.9 320 1618.9 320Q1618.9 320 1618.9 320H1210.9Q1210.9 320 1210.9 320Q1210.9 320 1210.9 320L1124.2 728Q1124.2 728 1124.2 728Q1124.2 728 1124.2 728ZM2375 884 2218.9 635H2421.9L2546.3 831L2535 884ZM1816 884 1968.8 165H2539.8Q2580.3 165 2609.8 185Q2639.3 205 2652.6 239Q2665.9 273 2657.4 313L2617.8 499Q2609.3 539 2581.6 573Q2553.9 607 2515.9 627Q2477.9 647 2437.4 647L2021.2 648L1971 884ZM2054.5 491H2462.5Q2462.5 491 2462.5 491Q2462.5 491 2462.5 491L2498.9 320Q2498.9 320 2498.9 320Q2498.9 320 2498.9 320H2090.9Q2090.9 320 2090.9 320Q2090.9 320 2090.9 320L2054.5 491Q2054.5 491 2054.5 491Q2054.5 491 2054.5 491Z",
  corsa: "M2959 884 3016.2 615 2779 164H2964L3129.7 448L3415 164H3602L3172 616L3115 884ZM3537 884 3548.3 831 3870.5 524 3678.8 217 3690 164H3851L4000.7 401L4249 164H4410L4398.8 217L4075.5 524L4268.5 830L4257 884H4096L3946.2 648L3698 884Z",
} as const;
// </wordmark:generated>

export type LogoTone = "on-dark" | "on-light" | "mono";

/** Wordmark colours per tone (mono draws in the current text colour) */
export const WORDMARK_TONES: Record<LogoTone, { libre: string; corsa: string }> = {
  "on-dark": { libre: LOGO_COLORS.chalk, corsa: LOGO_COLORS.volt },
  "on-light": { libre: LOGO_COLORS.void, corsa: LOGO_COLORS.volt },
  mono: { libre: "currentColor", corsa: "currentColor" },
};

/** The three bars of the full lock-up, in CSS px at the large size: volt, cyan, crimson, slanted like the rest */
export const BARS = {
  width: 16,
  height: 5,
  gap: 3,
  colors: [LOGO_COLORS.volt, LOGO_COLORS.cyan, LOGO_COLORS.crimson],
} as const;

/** The one tagline */
export const LOGO_TAGLINE = "Open-source 500 Hz motorsport sim";

/**
 * Lock-up proportions in CSS px, as the component draws them. `compact` is the small size (the one most screens use),
 * `full` the large size at its widest. The exported files are drawn at these proportions.
 */
export const LOCKUP = {
  compact: { mark: 32, word: 18, gap: 10 },
  full: {
    mark: 64,
    word: 41.6,
    gap: 16,
    /** space between the wordmark and the bars row */
    rowGap: 8,
    /** space between the bars and the tagline */
    barsToTagline: 8,
    tagline: { size: 11, tracking: 0.24, lineHeight: 1.5 },
  },
} as const;

/** Clear space round any lock-up: half the emblem's width. Smallest sizes: emblem 16 px, compact lock-up 96 px wide. */
export const CLEAR_SPACE_RATIO = 0.5;
export const MIN_EMBLEM_PX = 16;
export const MIN_COMPACT_WIDTH_PX = 96;
