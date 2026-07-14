import type { ParamSpec } from "../types";

interface Props {
  spec: ParamSpec;
  value: number | string | boolean | undefined;
  onChange: (value: number | string | boolean | null) => void;
}

/** One schema-driven form row. An empty value means "use the script's
 * default" — only overrides are stored in graph.json. */
export default function ParamField({ spec, value, onChange }: Props) {
  const modified = value !== undefined;
  const cls = modified ? "modified" : "";

  let control: React.ReactNode;
  if (spec.type === "flag") {
    control = (
      <input
        type="checkbox"
        checked={Boolean(value ?? spec.default)}
        onChange={(e) => onChange(e.target.checked ? true : null)}
      />
    );
  } else if (spec.choices) {
    control = (
      <select
        className={cls}
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">
          {spec.default != null ? `default (${spec.default})` : "—"}
        </option>
        {spec.choices.map((c) => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>
    );
  } else if (spec.type === "int" || spec.type === "float") {
    control = (
      <input
        className={cls}
        type="number"
        step={spec.type === "int" ? 1 : "any"}
        placeholder={spec.default != null ? String(spec.default) : ""}
        value={value === undefined ? "" : String(value)}
        onChange={(e) => {
          const v = e.target.value;
          if (v === "") return onChange(null);
          onChange(spec.type === "int" ? parseInt(v, 10) : parseFloat(v));
        }}
      />
    );
  } else {
    control = (
      <input
        className={cls}
        type="text"
        placeholder={spec.default != null ? String(spec.default) : ""}
        value={value === undefined ? "" : String(value)}
        onChange={(e) => onChange(e.target.value || null)}
      />
    );
  }

  return (
    <div className="param-row" title={`${spec.flag}\n${spec.help}`}>
      <label>{spec.name}{spec.required ? " *" : ""}</label>
      {control}
    </div>
  );
}
