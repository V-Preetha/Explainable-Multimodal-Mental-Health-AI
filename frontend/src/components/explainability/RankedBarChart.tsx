import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from "recharts";
import { EmptyState } from "@/components/ui/EmptyState";
import { BarChart3 } from "lucide-react";

interface RankedBarChartProps {
  data?: Record<string, number>;
  emptyTitle: string;
  emptyDescription: string;
  color?: string;
  height?: number;
}

export function RankedBarChart({ data, emptyTitle, emptyDescription, color = "#2f7d72", height = 260 }: RankedBarChartProps) {
  if (!data || Object.keys(data).length === 0) {
    return <EmptyState icon={<BarChart3 size={30} strokeWidth={1.25} />} title={emptyTitle} description={emptyDescription} />;
  }

  const rows = Object.entries(data)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#eef0f3" />
          <XAxis type="number" tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 11, fill: "#475569" }} axisLine={false} tickLine={false} />
          <RechartsTooltip
            cursor={{ fill: "rgba(47,125,114,0.06)" }}
            contentStyle={{ borderRadius: 10, border: "1px solid #e3e6ea", fontSize: 12 }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={18}>
            {rows.map((row) => (
              <Cell key={row.name} fill={color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
