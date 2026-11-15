import type { ReactNode } from "react";
import {
  datasets,
  deviations,
  hypotheses,
  limitations,
  method,
  metrics,
  plan,
  repo,
  sources,
} from "@/lib/content";

const prose = "max-w-[62ch] text-lg leading-relaxed";
// The site's one underline treatment. Two layers: `before` is a 1px hairline at
// 30% opacity, always on, so the link reads as a link at rest; `after` is the 2px
// rule that draws left→right on hover/focus and retracts right→left.
// Tailwind v4 emits `scale-*` as the standalone `scale` property, not
// `transform`, so the transition is `transition-[scale]` — `transition-transform`
// would animate nothing. Origin flips at the instant scale is 0, so it is
// invisible: at rest right (the rule shrinks away to the right), on hover left
// (it grows from the left).
const link =
  "relative inline-block before:absolute before:inset-x-0 before:bottom-0 before:h-px before:bg-current before:opacity-30 before:content-[''] after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:origin-right after:scale-x-0 after:bg-current after:content-[''] after:transition-[scale] after:duration-[250ms] after:ease-out hover:after:origin-left hover:after:scale-x-100 focus-visible:after:origin-left focus-visible:after:scale-x-100 motion-reduce:after:transition-none";

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-16 border-t border-foreground/10">
      <div className="mx-auto grid max-w-5xl gap-4 px-6 py-14 sm:grid-cols-[11rem_1fr] sm:gap-10">
        <h2 className="font-sans text-sm text-foreground/50">{title}</h2>
        <div className="flex min-w-0 flex-col gap-6">{children}</div>
      </div>
    </section>
  );
}

function Table({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[34rem] border-collapse text-left font-sans text-sm tabular-nums">
        <thead>
          <tr className="border-b border-foreground/20 text-foreground/50">
            {head.map((h) => (
              <th key={h} className="py-2 pr-4 font-normal">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-foreground/10 align-top">
              {r.map((c, j) => (
                <td key={j} className="py-3 pr-4 leading-relaxed">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Every list on the page uses this marker: a small dot in a fixed gutter. It is
// a span rather than the `list-disc` marker so size, colour and alignment stay
// identical across rows whose text is `text-lg` and rows that are `text-sm`
// (browser markers inherit the text size and sit differently in each), and so
// the gutter is a real grid column instead of the marker box's own width.
//
// `offset` is a single class, not a base class plus an override: two
// `mt-[…]` utilities in one class list would resolve by stylesheet order, not
// by the order they appear in the attribute, so the alignment would be a
// coin-flip. Exactly one margin utility is emitted per row instead.
function Dot({ offset = "mt-[0.7rem]" }: { offset?: string }) {
  return (
    <span
      aria-hidden="true"
      className={`${offset} h-1.5 w-1.5 shrink-0 rounded-full bg-foreground/30`}
    />
  );
}

function Rows({ items }: { items: [string, string][] }) {
  return (
    <ul className="divide-y divide-foreground/10 border-y border-foreground/10">
      {items.map(([title, text]) => (
        <li key={title} className="grid grid-cols-[1.25rem_1fr] gap-3 py-4">
          {/* `mt-[0.45rem]` centres the 6px dot on the `text-sm` title's first
              line; the larger bullet rows use the Dot default for `text-lg`. */}
          <Dot offset="mt-[0.45rem]" />
          <div>
            <p className="font-sans text-sm font-medium">{title}</p>
            <p className="mt-1 max-w-[62ch] leading-relaxed text-foreground/70">
              {text}
            </p>
          </div>
        </li>
      ))}
    </ul>
  );
}

function Bullets({
  items,
  textClassName,
  dotOffset,
}: {
  items: string[];
  textClassName: string;
  dotOffset?: string;
}) {
  return (
    <ul className="divide-y divide-foreground/10 border-y border-foreground/10">
      {items.map((text) => (
        <li key={text} className="grid grid-cols-[1.25rem_1fr] gap-3 py-4">
          <Dot offset={dotOffset} />
          <span className={textClassName}>{text}</span>
        </li>
      ))}
    </ul>
  );
}

export default function Report() {
  return (
    <>
      <Section id="question" title="question">
        <p className={prose}>
          i trained a small lora adapter on 15 to 30 minutes of audio and asked
          whether it closes whisper-small&apos;s gap on yoruba, and what it
          costs on other speech. yoruba is my main language axis. accent is a
          second axis, which i pool only.
        </p>
        <p className={prose}>
          this is my report. every number comes with its interval and sample
          size, and failures get the same space as results.
        </p>
        <Bullets
          items={hypotheses.map(([, text]) => text)}
          textClassName="max-w-[62ch] leading-relaxed"
          dotOffset="mt-[0.6rem]"
        />
      </Section>

      <Section id="data" title="data">
        <Table
          head={["group", "role", "clips (speakers)", "licence"]}
          rows={datasets}
        />
        <p className={prose}>
          i capped each split and filled it with the first rows in file order,
          so my splits are not random (about 64, 15 and 21 percent). i publish
          no audio.
        </p>
      </Section>

      <Section id="method" title="method">
        {method.map((p) => (
          <p key={p} className={prose}>
            {p}
          </p>
        ))}
        <Table head={["metric", "status", "note"]} rows={metrics} />
      </Section>

      <Section id="results" title="results">
        <p className={prose}>i will put the numbers here once i freeze the runs.</p>
      </Section>

      <Section id="deviations" title="deviations">
        <p className={prose}>
          every change to the plan is a dated row in the{" "}
          <a href={plan} target="_blank" rel="noreferrer" className={link}>
            deviations log
          </a>
          . these are the ones that matter.
        </p>
        <Rows items={deviations} />
      </Section>

      <Section id="limitations" title="limitations">
        <Bullets
          items={limitations}
          textClassName={`${prose} text-foreground/80`}
        />
      </Section>

      <Section id="sources" title="sources">
        <Bullets
          items={sources}
          textClassName="font-sans text-sm text-foreground/80"
          dotOffset="mt-[0.45rem]"
        />
        <p className={prose}>
          code, caches and the full log are in the{" "}
          <a href={repo} target="_blank" rel="noreferrer" className={link}>
            repo
          </a>
          .
        </p>
      </Section>
    </>
  );
}