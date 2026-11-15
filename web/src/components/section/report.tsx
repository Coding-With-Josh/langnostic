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
import { Highlighter } from "@/components/ui/highlighter";

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

// One colour, one meaning. If a word does not fall in a category below it stays
// plain — decoration is what made the first pass unreadable.
const ORANGE = "#FF9D00";       // key subject or identifier: model, method, language, commit
const LIGHT_BLUE = "#93C5FD";   // an important number: budget, error rate
const LIGHT_RED = "#FCA5A5";    // a warning: a test that failed, a result that cannot be read
const LIGHT_GREEN = "#86EFAC";  // a check that passed or a failure that was recovered
const LIGHT_PURPLE = "#C4B5FD"; // a dataset name

type Action =
  | "highlight"
  | "underline"
  | "box"
  | "circle"
  | "strike-through"
  | "crossed-off"
  | "bracket";

type Terms = Record<string, { color: string; action?: Action }>;

// Splits `text` on every occurrence of each `needle` and wraps the match in a
// Highlighter, leaving the rest of the string as plain text. Used to inject
// inline annotations into strings that come from content.ts as plain strings.
//
// Overlap precedence is deterministic rather than incidental: matches sort by
// index, then by length descending, so at equal start the longest needle wins
// (`whisper-small` beats `whisper`) and the loser is dropped entirely. Sorting
// by index alone would let JS key-insertion order decide which annotation
// renders. Phase 3, substring collision.
function mark(text: string, terms: Terms): ReactNode {
  const matches: { index: number; length: number; needle: string }[] = [];
  for (const needle of Object.keys(terms)) {
    let idx = text.indexOf(needle);
    while (idx !== -1) {
      matches.push({ index: idx, length: needle.length, needle });
      idx = text.indexOf(needle, idx + 1);
    }
  }
  matches.sort((a, b) => a.index - b.index || b.length - a.length);

  // Drop overlapping matches: "whisper" would otherwise match inside
  // "whisper-small" and produce two stacked annotations on one word.
  const nonOverlapping: typeof matches = [];
  let lastEnd = -1;
  for (const m of matches) {
    if (m.index >= lastEnd) {
      nonOverlapping.push(m);
      lastEnd = m.index + m.length;
    }
  }

  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of nonOverlapping) {
    if (m.index > last) {
      parts.push(text.slice(last, m.index));
    }
    const { color, action = "circle" } = terms[m.needle];
    parts.push(
      <Highlighter key={`${m.needle}-${m.index}`} action={action} color={color}>
        {m.needle}
      </Highlighter>,
    );
    last = m.index + m.length;
  }
  if (last < text.length) {
    parts.push(text.slice(last));
  }
  return <>{parts}</>;
}

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
    <section
      id={id}
      className="scroll-mt-16 border-t border-black/10 text-black"
    >
      <div className="mx-auto grid max-w-5xl gap-4 px-6 py-14 sm:grid-cols-[11rem_1fr] sm:gap-10">
        <h2 className="font-sans text-sm text-black/50">{title}</h2>
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
          <tr className="border-b border-black/20 text-black/50">
            {head.map((h) => (
              <th key={h} className="py-2 pr-4 font-normal">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-black/10 align-top">
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
      className={`${offset} h-1.5 w-1.5 shrink-0 rounded-full bg-black/30`}
    />
  );
}

// The title and the body are different strings and are annotated separately.
// Every "failed / bad / quota" needle lives in the *title* (`deviations[n][0]`),
// which is why the first pass — which only ran mark() over the body — put zero
// highlights in this section. Phase 2, verbatim-needle assumption.
const deviationTitleTerms: Terms = {
  "bad test": { color: LIGHT_RED, action: "highlight" },
  "failed its gates": { color: LIGHT_RED, action: "highlight" },
  "failed twice": { color: LIGHT_RED, action: "highlight" },
  "colab ran out of my gpu quota": { color: LIGHT_RED, action: "highlight" },
};

const deviationTextTerms: Terms = {
  // identifiers: the commit that shipped the bad test, and the one that fixed it
  "commit 7016051": { color: ORANGE, action: "circle" },
  "08dfe48": { color: LIGHT_GREEN, action: "circle" },
  // the error rates themselves, not a description of them
  "142.4%": { color: LIGHT_BLUE, action: "underline" },
  "127.5%": { color: LIGHT_BLUE, action: "underline" },
  "97.13%": { color: LIGHT_BLUE, action: "underline" },
  "131.6%": { color: LIGHT_BLUE, action: "underline" },
  // why the validation and test numbers above cannot be compared
  "without chunking": { color: LIGHT_RED, action: "highlight" },
  // a check that passed after the session reset
  "manifest hashes matched": { color: LIGHT_GREEN, action: "underline" },
  // the two datasets whose group names collide
  naijavoices: { color: LIGHT_PURPLE, action: "circle" },
  voxpopuli: { color: LIGHT_PURPLE, action: "circle" },
};

function Rows({ items }: { items: [string, string][] }) {
  return (
    <ul className="divide-y divide-black/10 border-y border-black/10">
      {items.map(([title, text]) => (
        <li key={title} className="grid grid-cols-[1.25rem_1fr] gap-3 py-4">
          {/* `mt-[0.45rem]` centres the 6px dot on the `text-sm` title's first
              line; the larger bullet rows use the Dot default for `text-lg`. */}
          <Dot offset="mt-[0.45rem]" />
          <div>
            <p className="font-sans text-sm font-medium">
              {mark(title, deviationTitleTerms)}
            </p>
            <p className="mt-1 max-w-[62ch] leading-relaxed text-black/70">
              {mark(text, deviationTextTerms)}
            </p>
          </div>
        </li>
      ))}
    </ul>
  );
}

// Term sets are per-list, not shared. A shared set leaks across corpora: with
// one set for hypotheses + limitations + sources, `adapter` landed on two
// limitation rows where it is incidental, and `lora` / `15 and 30 minute`
// matched nothing at all. Phase 3, wrong-field matching.
//
// The hypotheses are eight predictions, so there is no measured value and no
// failure inside them to flag — circling `adapter` and `Yoruba` there only
// repeated nouns the paragraph above already carried. The one thing a reader
// must not misread is that h1 is descriptive, not a test, so that is the mark.
const hypothesisTerms: Terms = {
  "descriptive only": { color: LIGHT_RED, action: "highlight" },
};

const limitationTerms: Terms = {
  // the four limits that change how the report should be read
  "one seed per run": { color: LIGHT_RED, action: "highlight" },
  "cannot separate": { color: LIGHT_RED, action: "highlight" },
  "no live demo": { color: LIGHT_RED, action: "highlight" },
  "unofficial mirror": { color: LIGHT_RED, action: "highlight" },
};

function Bullets({
  items,
  textClassName,
  dotOffset,
  terms,
}: {
  items: string[];
  textClassName: string;
  dotOffset?: string;
  terms?: Terms;
}) {
  return (
    <ul className="divide-y divide-black/10 border-y border-black/10">
      {items.map((text) => (
        <li key={text} className="grid grid-cols-[1.25rem_1fr] gap-3 py-4">
          <Dot offset={dotOffset} />
          <span className={textClassName}>{mark(text, terms ?? {})}</span>
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
          i trained a small <Highlighter action="highlight" color="#14EFFF">
            lora adapter
          </Highlighter> on 15 to 30 minutes of audio and asked whether it
          closes <span className="italic">
            whisper-small&apos;s
          </span> gap on{" "}
          <span className="font-medium">
            Yoruba
          </span>, and
          what it costs on other speech. Yoruba is my main language axis.
          accent is a second axis, which i pool only.
        </p>
        <p className={prose}>
          this is my report. every number comes with its interval and sample
          size, and failures get the same space as results.
        </p>
        <Bullets
          items={hypotheses.map(([, text]) => text)}
          textClassName="max-w-[62ch] leading-relaxed"
          dotOffset="mt-[0.6rem]"
          terms={hypothesisTerms}
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
            {mark(p, {
              "whisper-small": { color: ORANGE, action: "circle" },
              lora: { color: ORANGE, action: "circle" },
              "15 and 30 minute": { color: LIGHT_BLUE, action: "box" },
            })}
          </p>
        ))}
        <Table head={["metric", "status", "note"]} rows={metrics} />
      </Section>

      <Section id="results" title="results">
        <p className={prose}>
          i will put the numbers here once i freeze the runs.
        </p>
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
          textClassName={`${prose} text-black/80`}
          terms={limitationTerms}
        />
      </Section>

      <Section id="sources" title="sources">
        {/* Sources are names and links: no number, no warning, nothing to mark. */}
        <Bullets
          items={sources}
          textClassName="font-sans text-sm text-black/80"
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
