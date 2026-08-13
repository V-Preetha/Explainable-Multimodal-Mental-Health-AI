import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

export function formatPercent(value: number | undefined, digits = 0): string {
  if (value === undefined || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value: number | undefined, digits = 1): string {
  if (value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
