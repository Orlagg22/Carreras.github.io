/**
 * The soundtrack: original tracks synthesised from scratch (no samples, no recordings: free of third-party copyright,
 * released as CC0), one MP3 each under /music/, loudness-matched to -16 LUFS. Each track belongs to one pool, and the music director picks the pool from where the
 * player is (music-director.ts): the menus, a race, or the results and replays.
 */
import { withBasePath } from "@/lib/base-path";

export type MusicPool = "menu" | "race" | "results";

export interface MusicTrack {
  /** The file's name under /music/ and the id the settings keep */
  id: string;
  title: string;
  composer: string;
  performer: string;
  pool: MusicPool;
  licence: "CC0" | "Public domain";
  /** Where the recording comes from */
  source: string;
  /** One of these opens the menus the first time they play */
  opener?: boolean;
}

const SOURCE = "Generated for Noryx (scripts/generate-soundtrack.py)";

export const MUSIC_TRACKS: readonly MusicTrack[] = [
  // The menus: one pool for every menu screen, so a track plays on across them
  { id: "pit-lane-dawn", title: "Pit Lane Dawn", composer: "Noryx", performer: "Synthesized", pool: "menu", licence: "CC0", source: SOURCE, opener: true },
  { id: "quiet-paddock", title: "Quiet Paddock", composer: "Noryx", performer: "Synthesized", pool: "menu", licence: "CC0", source: SOURCE },
  { id: "garage-light", title: "Garage Light", composer: "Noryx", performer: "Synthesized", pool: "menu", licence: "CC0", source: SOURCE, opener: true },
  { id: "slow-out-lap", title: "Slow Out Lap", composer: "Noryx", performer: "Synthesized", pool: "menu", licence: "CC0", source: SOURCE },

  // Racing: from the lights out (or the first metres of a lap) to the flag
  { id: "redline", title: "Redline", composer: "Noryx", performer: "Synthesized", pool: "race", licence: "CC0", source: SOURCE },
  { id: "apex-hunter", title: "Apex Hunter", composer: "Noryx", performer: "Synthesized", pool: "race", licence: "CC0", source: SOURCE },
  { id: "slipstream", title: "Slipstream", composer: "Noryx", performer: "Synthesized", pool: "race", licence: "CC0", source: SOURCE },
  { id: "turbo-pulse", title: "Turbo Pulse", composer: "Noryx", performer: "Synthesized", pool: "race", licence: "CC0", source: SOURCE },
  { id: "chicane", title: "Chicane", composer: "Noryx", performer: "Synthesized", pool: "race", licence: "CC0", source: SOURCE },
  { id: "full-throttle", title: "Full Throttle", composer: "Noryx", performer: "Synthesized", pool: "race", licence: "CC0", source: SOURCE },
  { id: "late-braking", title: "Late Braking", composer: "Noryx", performer: "Synthesized", pool: "race", licence: "CC0", source: SOURCE },

  // The flag, the cool-down lap, the podium, the results and the replays
  { id: "podium-glow", title: "Podium Glow", composer: "Noryx", performer: "Synthesized", pool: "results", licence: "CC0", source: SOURCE },
  { id: "checkered-flag", title: "Checkered Flag", composer: "Noryx", performer: "Synthesized", pool: "results", licence: "CC0", source: SOURCE },
  { id: "cool-down-lap", title: "Cool Down Lap", composer: "Noryx", performer: "Synthesized", pool: "results", licence: "CC0", source: SOURCE },
];

export const MUSIC_TRACK_IDS: ReadonlySet<string> = new Set(MUSIC_TRACKS.map((t) => t.id));

export const musicTrack = (id: string): MusicTrack | undefined => MUSIC_TRACKS.find((t) => t.id === id);

export const musicUrl = (track: MusicTrack): string => withBasePath(`/music/${track.id}.mp3`);

export const POOL_LABEL: Record<MusicPool, string> = {
  menu: "Menu",
  race: "Race",
  results: "Results and replay",
};
