"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BASE_ALTITUDE_M, type BuiltTrackData, DEFAULT_TRACK_ID, TURN_METADATA } from "@/lib/track-data";
import { getTrackInfo } from "@/lib/track-registry";
import { formatMegabytes, useAssetProgress, type AssetTask } from "@/lib/loading/asset-progress";
import { connectedPads } from "@/lib/input/device-input";
import { keyLabel } from "@/lib/input/bindings";
import { useKeyLayout } from "@/lib/input/use-key-layout";
import { enterFullscreenOnTouch } from "@/lib/install-app";
import { padPressTracker } from "./useMenuNav";
import { NoryxLogo } from "@/components/ui/NoryxLogo";
import { Meter } from "@/components/ui";
import { Icon } from "@/components/ui/icons";
import { PadGlyph } from "@/components/ui/prompts";
import { useInputMode } from "./input-mode";
import { useUnits } from "@/lib/settings/ui-prefs";
import { DISTANCE_LABEL, distanceIn } from "@/lib/settings/units";
import { carById, type CarId } from "@/lib/cars/car-registry";
import { carMaker, circuitMaker } from "@/lib/modders";

/** The tips, naming the player's own camera and racing-line keys as printed on their keyboard */
const tipsFor = (camera: string, line: string) => [
  "Esc pauses: the Garage, corner practice, AI races and multiplayer are all there.",
  "Brake in a straight line, then trail off the pedal as you turn into the apex.",
  `Press ${camera} to cycle cameras: cockpit, T-cam, chase and trackside TV.`,
  `The racing line turns red where you need to brake. Press ${line} to hide it once you know the circuit.`,
  "On a controller, the steering assist keeps the front tyres at their grip limit.",
  "Use the boost on the straights; the battery recharges under braking.",
  "Switch between open-wheelers, GT3s and road cars from the Car menu.",
];

/** A tip for a circuit high enough for the thin air to matter (to the nearest 50 m) */
const altitudeTip = (m: number) =>
  `This circuit sits ${(Math.round(m / 50) * 50).toLocaleString("en-US")} m above sea level: thinner air means less downforce and less drag than you expect.`;

const LIGHTS = 5;

/** One line of status: what is arriving now, with its size where it is known */
function statusLine(tasks: readonly AssetTask[]): string {
  const now = tasks.find((t) => t.status === "loading") ?? tasks.find((t) => t.status === "pending");
  if (!now) return "Everything is in";
  if (now.id === "shaders") return now.status === "loading" ? "Preparing the shaders" : "Waiting to prepare the shaders";
  if (now.status === "pending") return `${now.label}: queued`;
  return now.totalBytes > 0
    ? `${now.label}: ${formatMegabytes(now.loadedBytes)} of ${formatMegabytes(now.totalBytes)} MB`
    : `${now.label}: loading`;
}

/** Shown once loaded; touch screens are told to tap */
const ContinuePrompt: React.FC = () => {
  const { family } = useInputMode();
  return (
    <div className="flex flex-col gap-2">
      <span className="lc-disp lc-breathe text-[24px] text-lc-volt sm:text-[30px] phone:text-[22px]">
        <span className="pointer-coarse:hidden">Press any key to enter the track</span>
        <span className="hidden pointer-coarse:inline">Tap to enter the track</span>
      </span>
      <span className="lc-hint inline-flex flex-wrap items-center gap-1.5">
        <span className="pointer-coarse:hidden">A key, a click,</span>
        <span className="hidden pointer-coarse:inline">A tap, a key,</span>
        <PadGlyph button="a" family={family} />
        or any wheel button
      </span>
    </div>
  );
};

/**
 * Who made what is about to load: the circuit's modder, the car's (in its credit's words: the Liana's is its
 * physics), and the game. A maker who takes donations says so, and where: the loading screen itself takes no clicks
 * but the one that enters the track, so the link is the circuit's and car's Source button.
 */
const MakersLines: React.FC<{ trackId: string; trackName: string; carId: CarId }> = ({ trackId, trackName, carId }) => {
  const circuit = circuitMaker(trackId);
  const car = carMaker(carId);
  const lines = [
    circuit ? { key: "circuit", lead: `${trackName} by`, maker: circuit } : null,
    { key: "car", lead: `${carById(carId).title}: ${car.role.toLowerCase()} by`, maker: car },
    { key: "game", lead: "Noryx by", maker: { role: "Game", name: "Orlando" } },
  ].filter((l): l is NonNullable<typeof l> => l !== null);

  return (
    <div className="flex flex-col gap-2 border-t border-lc-line pt-3.5 short:pt-2.5">
      <ul className="m-0 flex list-none flex-col gap-1 p-0 text-[14px] leading-snug text-lc-slate short:text-[12px]">
        {lines.map((l) => (
          <li key={l.key}>
            {l.lead} <span className="font-semibold text-lc-chalk">{l.maker.name}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

/** One start-light column (two lamps) inside the pill */
