"""Hold the Textual stylesheet and the geometry constants it interpolates."""

from agentperf_local.tui.branding import AA_BRAND_CSS_VARIABLES
from agentperf_local.tui.widgets import chart_width

# The metrics column is exactly as wide as its charts plus the panel padding and its rule.
RUN_METRICS_COLUMN_PADDING = 2
RUN_METRICS_COLUMN_RULE_WIDTH = 1
# The column scrolls once the details view overflows a short terminal; a reserved
# one-cell gutter keeps the charts at full width whether or not the bar is showing.
RUN_METRICS_COLUMN_SCROLLBAR_WIDTH = 1
RUN_METRICS_COLUMN_WIDTH = (
    chart_width() + 2 * RUN_METRICS_COLUMN_PADDING + RUN_METRICS_COLUMN_RULE_WIDTH + RUN_METRICS_COLUMN_SCROLLBAR_WIDTH
)
# Setup-form geometry. The label column and value cap are the two chosen numbers; the
# status indent and its wrap width are derived from them so the columns cannot drift.
FIELD_LABEL_WIDTH = 18
FIELD_LABEL_GAP = 1
FIELD_VALUE_MAX_WIDTH = 60
FIELD_STATUS_INDENT = FIELD_LABEL_WIDTH + FIELD_LABEL_GAP
FIELD_STATUS_MAX_WIDTH = FIELD_STATUS_INDENT + FIELD_VALUE_MAX_WIDTH
RESULT_CHART_GAP = 2
# Three charts, each with its gap, plus the page's two-cell padding on either side.
RESULT_CHARTS_MIN_WIDTH = 3 * (chart_width() + RESULT_CHART_GAP) + 4

# Every hex value lives in tui.branding, next to a comment naming its brand-kit source.
APP_CSS = (
    AA_BRAND_CSS_VARIABLES
    + f"""
$field-label-width: {FIELD_LABEL_WIDTH};
$field-label-gap: {FIELD_LABEL_GAP};
$field-value-max-width: {FIELD_VALUE_MAX_WIDTH};
$field-status-indent: {FIELD_STATUS_INDENT};
$field-status-max-width: {FIELD_STATUS_MAX_WIDTH};
$run-metrics-column-width: {RUN_METRICS_COLUMN_WIDTH};
$run-metrics-column-padding: {RUN_METRICS_COLUMN_PADDING};
$result-chart-gap: {RESULT_CHART_GAP};
"""
    + """
Screen {
    background: $aa-bg;
    color: $aa-text;
    align-horizontal: center;
}

Footer, FooterKey, FooterKey > .footer-key--key, FooterKey > .footer-key--description {
    background: $aa-bg;
    color: $aa-muted;
}

Footer {
    padding: 0 2;
}

#shell {
    height: 1fr;
    max-width: 118;
    width: 100%;
    align-horizontal: center;
}

ContentSwitcher {
    height: 1fr;
}

.page {
    height: 1fr;
    padding: 1 2 0 2;
}

#welcome-brand {
    height: auto;
    margin-bottom: 2;
}

#welcome-heading {
    height: auto;
    padding: 1 0 0 2;
}

#welcome-heading .hero, #welcome-heading .lede {
    margin-bottom: 0;
}

#welcome-logo {
    color: $aa-purple;
    width: auto;
    height: auto;
}

.welcome-choice {
    width: 100%;
    height: auto;
    min-width: 0;
    border: none;
    padding: 0 1;
    margin-bottom: 1;
    background: $aa-surface;
    color: $aa-text;
    text-align: left;
    content-align: left top;
    text-wrap: wrap;
    text-overflow: clip;
}

.welcome-choice:focus, .welcome-choice:hover {
    background: $aa-focus;
    color: $aa-purple-light;
}

#run-progress-row {
    height: 1;
    margin-top: 1;
}

#run-progress {
    width: 1fr;
    margin: 0;
}

#activity-progress {
    height: 1;
    width: 100%;
    margin: 0;
}

#run-progress Bar, #activity-progress Bar {
    width: 1fr;
}

#run-counters {
    width: auto;
    color: $aa-muted;
    padding-left: 2;
}

.page.compact #run-progress-row {
    height: auto;
    layout: vertical;
}

.page.compact #run-counters {
    padding-left: 0;
    height: auto;
    width: 100%;
}

#run-context {
    height: auto;
    margin-top: 1;
    margin-bottom: 0;
    color: $aa-muted;
}

#run-body {
    height: 1fr;
    margin-top: 1;
}

#run-left {
    width: 1fr;
    height: 1fr;
}


#run-left-title, #run-trend-title {
    margin-top: 0;
}

#run-left-switcher {
    height: 1fr;
}

/* ActivityLog's own stylesheet already sizes its inner RichLog; the shared rule
   below adds only the scrollbar theming both logs need. */
#run-server-log {
    height: 1fr;
    background: transparent;
    scrollbar-size: 1 1;
    padding: 0;
}

#run-server-log, ActivityLog > RichLog {
    overflow-y: auto;
    scrollbar-color: $aa-border;
    scrollbar-color-hover: $aa-purple-dim;
    scrollbar-color-active: $aa-purple;
    scrollbar-background: $aa-bg;
}

#run-server-log:focus, ActivityLog > RichLog:focus {
    background: transparent;
    background-tint: transparent;
}

#run-ttft-chart, #run-trend-title, #run-trend {
    display: none;
}

#run.show-details #run-ttft-chart, #run.show-details #run-trend-title, #run.show-details #run-trend {
    display: block;
}

#run.show-details.short-run #run-trend-title, #run.show-details.short-run #run-trend,
#run.show-details.shorter-run #run-ttft-chart {
    display: none;
}

#preflight-server-check {
    height: auto;
    margin-top: 1;
}

#run-metrics-column {
    width: $run-metrics-column-width;
    height: 1fr;
    padding: 0 $run-metrics-column-padding;
    border-left: solid $aa-border;
    scrollbar-size: 1 1;
    scrollbar-gutter: stable;
}

#run-throughput {
    color: $aa-lime;
    width: auto;
    height: auto;
}

#run-throughput-caption, #result-throughput-caption {
    color: $aa-muted;
    height: auto;
    margin-bottom: 1;
}

DistributionChart {
    height: auto;
    margin-bottom: 1;
}

#run-trend {
    width: 100%;
    height: 1;
    margin-bottom: 1;
}

Sparkline > .sparkline--min-color {
    color: $aa-purple-dim;
}

Sparkline > .sparkline--max-color {
    color: $aa-lime;
}

#result-charts {
    height: auto;
    margin-top: 1;
    margin-bottom: 1;
}

#result-charts DistributionChart {
    width: auto;
    margin-right: $result-chart-gap;
    margin-bottom: 0;
}

.page.compact #run-metrics-column {
    display: none;
}

.page.compact.show-details #run-metrics-column {
    display: block;
    width: 100%;
    padding: 0;
    border: none;
}

.page.compact.show-details #run-left {
    dock: bottom;
    width: 100%;
    height: 1;
}

.page.compact.show-details #run-left-title, .page.compact.show-details #activity-lines {
    display: none;
}

.page.compact #run-body {
    margin-top: 0;
}

#result-charts.-empty {
    display: none;
}

#result-details {
    height: auto;
}

.page.stack-charts #result-charts {
    layout: vertical;
}

.page.stack-charts #result-charts DistributionChart {
    margin-bottom: 1;
}

#result-throughput {
    color: $aa-lime;
    width: auto;
    height: auto;
    margin-top: 1;
}

#result-throughput.-muted {
    color: $aa-muted;
}

.eyebrow {
    color: $aa-muted;
    text-style: none;
    height: auto;
    margin-bottom: 0;
}

.hero {
    color: $aa-text-bright;
    text-style: bold;
    height: auto;
    margin-bottom: 1;
}

.page-heading, .page-heading-copy {
    height: auto;
}

.kitty {
    width: auto;
    height: auto;
    margin-left: 2;
    color: $aa-muted;
}

.lede {
    color: $aa-muted;
    height: auto;
    margin-bottom: 1;
}

.card, #submit-panel, .success-card {
    margin: 1 0;
    height: auto;
}

#submit-panel Checkbox {
    margin: 0;
}

#submit-notice {
    color: $aa-muted;
    height: auto;
    margin: 0;
    padding-left: 3;
}

.success-card {
    color: $aa-muted;
}

.error-card {
    background: transparent;
    border-left: solid $aa-red;
    color: $aa-red;
    padding: 0 1;
    height: auto;
    margin: 1 0;
}

.cancel-card {
    background: transparent;
    border-left: solid $aa-orange;
    color: $aa-orange;
    padding: 0 1;
    height: auto;
    margin: 1 0;
}

.status-row {
    height: auto;
    padding: 0;
    margin-bottom: 1;
}

.field-row {
    height: auto;
    margin-bottom: 1;
}

.field-row Label {
    width: $field-label-width;
    padding: 0 $field-label-gap 0 0;
    color: $aa-muted;
}

/* Values stop at a shared right edge, so dropdown arrows sit beside their
   values instead of drifting to the terminal edge on wide screens. */
.field-row Input {
    width: 1fr;
    max-width: $field-value-max-width;
    height: 1;
    padding: 0 1;
    border: none;
    background: $aa-surface;
}

.section-label {
    color: $aa-muted;
    text-style: bold;
    height: auto;
    margin-top: 1;
}

/* The managed status opens at the value column (label width plus its gap) and
   wraps at the same right edge as the capped fields above it. */
#managed-deployment-status {
    padding-left: $field-status-indent;
    max-width: $field-status-max-width;
    margin-bottom: 0;
}

.page.compact #managed-deployment-status {
    padding-left: 0;
    max-width: 100%;
}

.field-row Input:focus {
    border: none;
    background: $aa-focus;
}

#config .field-row {
    margin-bottom: 0;
}

#config .actions {
    margin-bottom: 0;
    margin-top: 0;
}

#config-selection {
    margin-bottom: 1;
}

#output-row {
    margin-top: 1;
}

/* A disclosure button opens its page's disclosed sections; an empty section stays hidden. */
.page .disclosed {
    display: none;
}

.page.-expanded .disclosed {
    display: block;
}

.page.-expanded .disclosed.-empty {
    display: none;
}

.field-row Select {
    width: 1fr;
    max-width: $field-value-max-width;
    height: 1;
}

.field-row SelectCurrent {
    height: 1;
    padding: 0 1;
    border: none;
    background: $aa-surface;
}

.field-row Select:focus > SelectCurrent {
    border: none;
    background: $aa-focus;
}

Input > .input--cursor {
    background: $aa-purple-light;
    color: $aa-bg;
}

Input > .input--selection {
    background: $aa-purple-dim;
}

Input > .input--placeholder {
    color: $aa-placeholder;
}

SelectOverlay {
    background: $aa-panel;
    border: tall $aa-border;
}

/* A consent line is never truncated: long labels wrap onto extra rows. */
Checkbox {
    width: 1fr;
    height: auto;
    background: transparent;
    text-wrap: wrap;
    text-overflow: clip;
}

Checkbox:focus {
    background: $aa-focus;
}

/* The off glyph matches its own background, so an unchecked box reads as empty. */
Checkbox > .toggle--button {
    background: $aa-surface;
    color: $aa-surface;
}

Checkbox.-on > .toggle--button {
    background: $aa-surface;
    color: $aa-lime;
}

Checkbox:focus > .toggle--label {
    background: $aa-focus;
    color: $aa-text-bright;
    text-style: bold;
}

.key-hint {
    color: $aa-muted;
    height: auto;
    margin-top: 1;
}

.page.compact .key-hint {
    margin-top: 0;
}

.actions {
    dock: bottom;
    background: $aa-bg;
    height: 1;
    margin-top: 1;
}

.actions Button:first-child {
    color: $aa-purple-light !important;
    text-style: bold;
}

#result-upload-busy {
    width: 1fr;
    height: 1;
}

.actions Button, #run-cancel, .disclosure {
    background: transparent !important;
    border: none !important;
    color: $aa-muted !important;
    height: 1;
    min-width: 0;
    width: auto;
    padding: 0;
    margin-right: 3;
    text-style: none;
}

.actions Button:focus, #run-cancel:focus, .disclosure:focus,
.actions Button:hover, #run-cancel:hover, .disclosure:hover {
    background: $aa-focus !important;
    color: $aa-purple-light !important;
}

.actions Button:focus, #run-cancel:focus, .disclosure:focus {
    text-style: bold;
}

.actions Button:hover, #run-cancel:hover, .disclosure:hover {
    text-style: underline;
}

.field-row Input:hover, .field-row Select:hover > SelectCurrent, Checkbox:hover {
    background: $aa-focus;
}

.disclosure {
    margin-top: 1;
    margin-right: 0;
}

.actions Button:disabled {
    background: transparent !important;
    color: $aa-disabled !important;
}

/* While disabled, this button's label carries live run state, so it stays readable. */
#run-cancel:disabled {
    background: transparent !important;
    color: $aa-muted !important;
}

#model-layout {
    height: 1fr;
}

#model-list {
    width: 43%;
    height: 1fr;
    background: transparent;
    border: none;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}

#model-detail-pane {
    width: 57%;
    height: 1fr;
    margin-left: 1;
    padding-left: 2;
    border-left: solid $aa-border;
    scrollbar-size: 1 1;
}

#model-detail {
    height: auto;
}

OptionList > .option-list--option-highlighted {
    background: $aa-focus;
    color: $aa-text-bright;
    text-style: bold;
}

ProgressBar {
    margin: 1 0;
}

Bar > .bar--bar {
    color: $aa-purple;
    background: $aa-surface;
}

Bar > .bar--indeterminate {
    color: $aa-purple;
    background: $aa-surface;
}

Bar > .bar--complete {
    color: $aa-lime;
    background: $aa-surface;
}

PercentageStatus {
    color: $aa-muted;
}

#small-terminal-warning {
    display: none;
    background: transparent;
    color: $aa-orange;
    border-left: solid $aa-orange;
    padding: 0 1;
    height: auto;
}

#model-layout.compact {
    layout: vertical;
    height: auto;
}

#model-layout.compact #model-list {
    width: 100%;
    height: 7;
}

#model-layout.compact #model-detail-pane {
    width: 100%;
    height: 8;
    margin-left: 0;
    padding-left: 0;
    padding-top: 1;
    border-left: none;
    border-top: solid $aa-border;
}

.page.compact .field-row Input, .page.compact .field-row Select {
    max-width: 100%;
}

.page.compact {
    padding: 0 1;
}

.page.compact .hero {
    margin-bottom: 0;
}

.page.compact .card {
    margin: 0;
}

.page.compact .actions {
    margin-top: 0;
}

.success-title {
    color: $aa-lime;
}

.page.compact #result-throughput {
    margin-top: 0;
}

.page.short {
    padding-top: 0;
}

.page.short #welcome-brand {
    margin-bottom: 0;
}

.page.short #welcome-heading {
    padding-top: 0;
    padding-left: 0;
}

.page.short #welcome-logo {
    display: none;
}

.page.short .hero, .page.short .lede, .page.short .welcome-choice {
    margin-bottom: 0;
}

/* The #model-layout prefix is what outranks the .compact heights above; do not drop it. */
.page.short #model-layout #model-list {
    height: 4;
}

.page.short #model-layout #model-detail-pane {
    height: 6;
}

.page.short #run-progress-row, .page.short #run-context {
    margin-top: 0;
}

.page.short #run-metrics {
    margin-bottom: 0;
}

.page.short .kitty {
    display: none;
}
"""
)
