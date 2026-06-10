import discord
from discord import app_commands
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import traceback
import subprocess
import sys
from matchups.matchups_ui import JumpToGamesButton, GameSelectView

from matchups.matchups_helpers import build_top20_embeds

from fuzzywuzzy import process  # Fuzzy matching for player names

# -----------------------------
# CONFIG
# -----------------------------
USER_ID = 764099035585576970
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "normalized"

# Live loop globals
live_task: asyncio.Task | None = None
LIVE_INTERVAL = 60  # seconds

# -----------------------------
# PIPELINE IMPORTS
# -----------------------------
from ingestion.mlb_ingest import run as ingest_run
from ingestion.normalize import run as normalize_run
from model.train_batter_props import run as train_batter_run
from model.win_probability import run as winprob_run
from model.score_batter_props import run as batterprops_run
from model.score_team_props import run as teamprops_run
from utils.performance_tracker import get_historical_performance

# -----------------------------
# TEAM LOGOS & COLORS
# -----------------------------
TEAM_LOGOS = {
    "Arizona Diamondbacks": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "Atlanta Braves": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",
    "Baltimore Orioles": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",
    "Boston Red Sox": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",
    "Chicago Cubs": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",
    "Chicago White Sox": "https://a.espncdn.com/i/teamlogos/mlb/500/chw.png",
    "Cincinnati Reds": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",
    "Cleveland Guardians": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",
    "Colorado Rockies": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",
    "Detroit Tigers": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",
    "Houston Astros": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",
    "Kansas City Royals": "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "Los Angeles Angels": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "Los Angeles Dodgers": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",
    "Miami Marlins": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",
    "Milwaukee Brewers": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",
    "Minnesota Twins": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",
    "New York Mets": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",
    "New York Yankees": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",
    "Oakland Athletics": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "Philadelphia Phillies": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",
    "Pittsburgh Pirates": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",
    "San Diego Padres": "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "San Francisco Giants": "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "Seattle Mariners": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",
    "St. Louis Cardinals": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",
    "Tampa Bay Rays": "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "Texas Rangers": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",
    "Toronto Blue Jays": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",
    "Washington Nationals": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",
}

TEAM_COLORS = {
    "Arizona Diamondbacks": ("#A71930", "#000000"),
    "Atlanta Braves": ("#CE1141", "#13274F"),
    "Baltimore Orioles": ("#DF4601", "#000000"),
    "Boston Red Sox": ("#BD3039", "#0D2B56"),
    "Chicago Cubs": ("#0E3386", "#CC3433"),
    "Chicago White Sox": ("#27251F", "#C4CED4"),
    "Cincinnati Reds": ("#C6011F", "#000000"),
    "Cleveland Guardians": ("#00385D", "#E50022"),
    "Colorado Rockies": ("#33006F", "#C4CED4"),
    "Detroit Tigers": ("#0C2340", "#FA4616"),
    "Houston Astros": ("#002D62", "#EB6E1F"),
    "Kansas City Royals": ("#004687", "#BD9B60"),
    "Los Angeles Angels": ("#BA0021", "#003263"),
    "Los Angeles Dodgers": ("#005A9C", "#EF3E42"),
    "Miami Marlins": ("#00A3E0", "#EF3340"),
    "Milwaukee Brewers": ("#12284B", "#FFC52F"),
    "Minnesota Twins": ("#002B5C", "#D31145"),
    "New York Mets": ("#002D72", "#FF5910"),
    "New York Yankees": ("#003087", "#E4002B"),
    "Oakland Athletics": ("#003831", "#EFB21E"),
    "Philadelphia Phillies": ("#E81828", "#002D72"),
    "Pittsburgh Pirates": ("#000000", "#FDB827"),
    "San Diego Padres": ("#2F241D", "#FFC425"),
    "San Francisco Giants": ("#FD5A1E", "#000000"),
    "Seattle Mariners": ("#0C2C56", "#005C5C"),
    "St. Louis Cardinals": ("#C41E3A", "#0A2252"),
    "Tampa Bay Rays": ("#092C5C", "#8FBCE6"),
    "Texas Rangers": ("#003278", "#C0111F"),
    "Toronto Blue Jays": ("#134A8E", "#1D2D5C"),
    "Washington Nationals": ("#AB0003", "#14225A"),
}

# -----------------------------
# INTENTS
# -----------------------------
intents = discord.Intents.default()
intents.messages = True
intents.dm_messages = True

# -----------------------------
# PAGINATOR VIEW
# -----------------------------

class Paginator(discord.ui.View):
    def __init__(self, embeds, labels=None, user_id=None):
        super().__init__(timeout=300)
        self.embeds = embeds
        self.index = 0
        self.user_id = user_id

        if labels is None:
            labels = [f"Page {i+1}" for i in range(len(embeds))]

        options = [
            discord.SelectOption(label=labels[i][:100], value=str(i))
            for i in range(len(embeds))
        ]

        self.select = discord.ui.Select(placeholder="Jump to page…", options=options)
        self.select.callback = self.on_select
        self.add_item(self.select)

    async def on_select(self, interaction):
        if self.user_id and interaction.user.id != self.user_id:
            await interaction.response.defer()
            return
        self.index = int(self.select.values[0])
        await interaction.response.edit_message(embed=self.embeds[self.index], view=self)

    @discord.ui.button(label="Prev", style=discord.ButtonStyle.secondary)
    async def prev(self, interaction, button):
        if self.user_id and interaction.user.id != self.user_id:
            await interaction.response.defer()
            return
        self.index = (self.index - 1) % len(self.embeds)
        await interaction.response.edit_message(embed=self.embeds[self.index], view=self)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.secondary)
    async def next(self, interaction, button):
        if self.user_id and interaction.user.id != self.user_id:
            await interaction.response.defer()
            return
        self.index = (self.index + 1) % len(self.embeds)
        await interaction.response.edit_message(embed=self.embeds[self.index], view=self)

# -----------------------------
# CLIENT
# -----------------------------
class MyClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def on_ready(self):
        print(f"[READY] Logged in as {self.user} (ID: {self.user.id})")
        synced = await self.tree.sync()
        print(f"[SYNC] Synced {len(synced)} global commands.")
        print("[READY] Bot is ready.")

client = MyClient()
tree = client.tree

# -----------------------------
# HELPERS
# -----------------------------
def player_headshot_url(player_id):
    return f"https://img.mlbstatic.com/mlb-photos/image/upload/w_600,q_100/v1/people/{player_id}/headshot/67/current"

def today_str():
    return str(date.today())

def fmt_prob(p):
    try:
        return f"{float(p) * 100:.1f}%"
    except:
        return "N/A"

def best_player_match(query, names):
    if not names:
        return None
    match, score = process.extractOne(query, names)
    return match if score >= 60 else None

def team_color(team_name):
    colors = TEAM_COLORS.get(team_name)
    if not colors:
        return 0x1E90FF
    return int(colors[0].replace("#", ""), 16)

def team_logo(team_name):
    return TEAM_LOGOS.get(team_name)

# -----------------------------
# LIVE INGESTION HELPERS
# -----------------------------
async def run_ingest_once(live=False):
    ingest_path = BASE_DIR / "ingestion" / "mlb_ingest.py"
    normalize_path = BASE_DIR / "ingestion" / "normalize.py"

    cmd = [sys.executable, str(ingest_path), "--date", today_str()]
    if live:
        cmd.append("--live")

    await asyncio.to_thread(subprocess.run, cmd, check=True)
    await asyncio.to_thread(subprocess.run, [sys.executable, str(normalize_path)], check=True)

async def live_loop(channel):
    global live_task
    try:
        while True:
            await run_ingest_once(live=True)

            games_path = DATA_DIR / "fact_games.parquet"
            if games_path.exists():
                gdf = pd.read_parquet(games_path)
                gdf["game_date"] = pd.to_datetime(gdf["game_date"]).dt.date
                today = date.today()
                today_games = gdf[gdf["game_date"] == today]

                lines = []
                in_progress = 0
                for _, r in today_games.iterrows():
                    away = r["away_team_name"]
                    home = r["home_team_name"]
                    away_score = r["away_score"]
                    home_score = r["home_score"]
                    status = r["status"]
                    lines.append(f"{away} {away_score} — {home} {home_score} ({status})")
                    if status not in ("Final", "Game Over"):
                        in_progress += 1

                desc = "\n".join(lines) if lines else "No games found."

                embed = discord.Embed(
                    title="Live Update — Ingestion Cycle Complete",
                    description=desc,
                    color=0x32CD32,
                )
                embed.set_footer(text=f"Games in progress: {in_progress} | Interval: {LIVE_INTERVAL}s")
                await channel.send(embed=embed)

            await asyncio.sleep(LIVE_INTERVAL)

    except asyncio.CancelledError:
        await channel.send("Live ingestion loop stopped.")
        live_task = None

# -----------------------------
# COMMAND: /ping
# -----------------------------
@tree.command(name="ping", description="Test if the bot is alive.")
async def ping_cmd(interaction):
    if interaction.user.id != USER_ID:
        return
    await interaction.response.send_message("Pong!", ephemeral=True)

# -----------------------------
# COMMAND: /run (FULL PIPELINE)
# -----------------------------
@tree.command(name="run", description="Run full MLB pipeline for today.")
@app_commands.describe(skip_training="Skip batter training if models are current (default: False)")
async def run_cmd(interaction, skip_training: bool = False):
    if interaction.user.id != USER_ID:
        return

    if skip_training:
        msg = "Starting pipeline for today (ingest → normalize → [skip train] → winprob → batter props → team props)…"
    else:
        msg = "Starting full pipeline for today (ingest → normalize → train → winprob → batter props → team props)…"

    await interaction.response.send_message(msg, ephemeral=True)

    try:
        game_date = today_str()

        await asyncio.to_thread(ingest_run, game_date, True)
        await asyncio.to_thread(normalize_run)
        if not skip_training:
            await asyncio.to_thread(train_batter_run)
        await asyncio.to_thread(winprob_run, game_date)
        await asyncio.to_thread(batterprops_run)
        await asyncio.to_thread(teamprops_run)

        if skip_training:
            await interaction.followup.send("Pipeline complete. All CSVs updated (training skipped).", ephemeral=True)
        else:
            await interaction.followup.send("Pipeline complete. All CSVs updated.", ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Pipeline failed: {e}", ephemeral=True)

# -----------------------------
# COMMAND: /runlive
# -----------------------------
@tree.command(name="runlive", description="Run one live ingestion cycle.")
async def runlive_cmd(interaction):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Running live ingestion…", ephemeral=True)

    try:
        await run_ingest_once(live=True)

        games_path = DATA_DIR / "fact_games.parquet"
        if games_path.exists():
            gdf = pd.read_parquet(games_path)
            gdf["game_date"] = pd.to_datetime(gdf["game_date"]).dt.date
            today = date.today()
            today_games = gdf[gdf["game_date"] == today]

            lines = []
            for _, r in today_games.iterrows():
                away = r["away_team_name"]
                home = r["home_team_name"]
                away_score = r["away_score"]
                home_score = r["home_score"]
                status = r["status"]
                lines.append(f"{away} {away_score} — {home} {home_score} ({status})")

            desc = "\n".join(lines) if lines else "No games found."

            embed = discord.Embed(
                title="Live Ingestion Complete",
                description=desc,
                color=0x00AAFF,
            )
            embed.set_footer(text=f"Date: {today}")
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.followup.send("Live ingestion complete, but fact_games.parquet not found.", ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Live ingestion failed: {e}", ephemeral=True)

# -----------------------------
# COMMAND: /runlive_loop
# -----------------------------
@tree.command(name="runlive_loop", description="Start live ingestion loop (every 60s).")
async def runlive_loop_cmd(interaction, stop: bool = False):
    global live_task

    if interaction.user.id != USER_ID:
        return

    if stop:
        if live_task and not live_task.done():
            live_task.cancel()
            await interaction.response.send_message("Stopping live ingestion loop…", ephemeral=True)
        else:
            await interaction.response.send_message("No live ingestion loop is running.", ephemeral=True)
        return

    if live_task and not live_task.done():
        await interaction.response.send_message("Live ingestion loop is already running.", ephemeral=True)
        return

    await interaction.response.send_message("Starting live ingestion loop (every 60s)…", ephemeral=True)
    live_task = asyncio.create_task(live_loop(interaction.channel))

# -----------------------------
# COMMAND: /stoplive
# -----------------------------
@tree.command(name="stoplive", description="Stop the live ingestion loop.")
async def stoplive_cmd(interaction):
    global live_task

    if interaction.user.id != USER_ID:
        return

    if live_task and not live_task.done():
        live_task.cancel()
        await interaction.response.send_message("Live ingestion loop stopped.", ephemeral=True)
    else:
        await interaction.response.send_message("No live ingestion loop is running.", ephemeral=True)

# -----------------------------
# COMMAND: /playerprops
# -----------------------------
@tree.command(name="playerprops", description="Show today's player props for a player.")
@app_commands.describe(player="Player name to search for", min_confidence="Minimum confidence threshold (0-100)", prop_type="Filter by prop type (hit, hr, multi, k)")
async def playerprops_cmd(interaction, player: str, min_confidence: int = None, prop_type: str = None):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading player props…", ephemeral=True)

    try:
        props_path = DATA_DIR / "fact_player_props.parquet"
        stats_path = DATA_DIR / "fact_player_stats.parquet"
        games_path = DATA_DIR / "fact_games.parquet"

        if not (props_path.exists() and stats_path.exists() and games_path.exists()):
            await interaction.followup.send("Missing props/stats/games files.", ephemeral=True)
            return

        props_df = pd.read_parquet(props_path)
        stats_df = pd.read_parquet(stats_path)
        games_df = pd.read_parquet(games_path)

        # Filter props & games to today
        today_date = date.today()
        props_df["game_date"] = pd.to_datetime(props_df["game_date"], errors="coerce").dt.date
        props_df = props_df[props_df["game_date"] == today_date]

        games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce").dt.date
        games_today = games_df[games_df["game_date"] == today_date]

        if props_df.empty:
            await interaction.followup.send("No props found for today.", ephemeral=True)
            return

        # Build name list for fuzzy match
        names = stats_df["player_name"].dropna().unique().tolist()
        best = best_player_match(player, names)

        if not best:
            await interaction.followup.send(f"No close match found for '{player}'.", ephemeral=True)
            return

        # All props rows for this player today
        player_rows = props_df[props_df["player_name"] == best]
        if player_rows.empty:
            await interaction.followup.send(f"No props found for {best}.", ephemeral=True)
            return

        # Apply confidence filtering
        if min_confidence is not None:
            min_conf = min_confidence / 100.0  # Convert to decimal
            filtered_rows = []
            for _, row in player_rows.iterrows():
                # Check if any prop meets the confidence threshold
                meets_threshold = False
                for prop in ['hit', 'hr', 'multi', 'k']:
                    prob_col = f'{prop}_prob'
                    if prob_col in row and row[prob_col] >= min_conf:
                        meets_threshold = True
                        break
                if meets_threshold:
                    filtered_rows.append(row)
            player_rows = pd.DataFrame(filtered_rows)
            
            if player_rows.empty:
                await interaction.followup.send(f"No props found for {best} with confidence >= {min_confidence}%.", ephemeral=True)
                return

        # Apply prop type filtering
        if prop_type is not None:
            prop_type = prop_type.lower()
            if prop_type not in ['hit', 'hr', 'multi', 'k']:
                await interaction.followup.send(f"Invalid prop_type. Use: hit, hr, multi, or k", ephemeral=True)
                return
            
            # Check if the player has this prop type with valid probability
            prob_col = f'{prop_type}_prob'
            if prob_col not in player_rows.columns or player_rows[prob_col].isna().all():
                await interaction.followup.send(f"No {prop_type} predictions found for {best}.", ephemeral=True)
                return

        # Stats rows for headshot (latest row)
        stats_rows = stats_df[stats_df["player_name"] == best].copy()
        if "season" in stats_rows.columns:
            stats_rows = stats_rows.sort_values("season", ascending=False)
        stats_row = stats_rows.iloc[0]
        player_id = stats_row.get("player_id")

        # Opposing pitcher lookup
        pitcher_stats = stats_df[stats_df["group"] == "pitching"]
        pitcher_lookup = dict(zip(pitcher_stats["player_id"], pitcher_stats["player_name"]))

        # Build (game_pk, team_id) → team_name lookup from fact_games
        team_lookup = {}
        for _, g in games_today.iterrows():
            team_lookup[(g["game_pk"], g["home_team_id"])] = g["home_team_name"]
            team_lookup[(g["game_pk"], g["away_team_id"])] = g["away_team_name"]

        embeds = []
        labels = []

        for _, row in player_rows.iterrows():
            game_date = row.get("game_date")
            game_pk = row.get("game_pk")
            team_id = row.get("team_id")

            # TRUST games_df for team name
            team_name = team_lookup.get((game_pk, team_id), row.get("team_name"))

            opp_pitcher_id = row.get("opp_pitcher_id")
            opp_pitcher_name = pitcher_lookup.get(opp_pitcher_id, "Unknown Pitcher")

            title = f"{best} – Player Props ({game_date})"
            desc_lines = []

            # Only show the requested prop type if specified
            if prop_type:
                prob_col = f'{prop_type}_prob'
                ci_lower_col = f'{prop_type}_prob_ci_lower'
                ci_upper_col = f'{prop_type}_prob_ci_upper'
                streak_col = f'{prop_type}_streak'
                
                if prob_col in row:
                    prob = row.get(prob_col)
                    ci_lower = row.get(ci_lower_col)
                    ci_upper = row.get(ci_upper_col)
                    if ci_lower and ci_upper:
                        desc_lines.append(f"**{prop_type.upper()} Prob:** {fmt_prob(prob)} ({fmt_prob(ci_lower)}-{fmt_prob(ci_upper)})")
                    else:
                        desc_lines.append(f"**{prop_type.upper()} Prob:** {fmt_prob(prob)}")
                if streak_col in row:
                    desc_lines.append(f"**{prop_type.upper()} Streak:** {row.get(streak_col)}")
            else:
                # Show all props
                if "hit_prob" in row:
                    hit_prob = row.get('hit_prob')
                    ci_lower = row.get('hit_prob_ci_lower')
                    ci_upper = row.get('hit_prob_ci_upper')
                    if ci_lower and ci_upper:
                        desc_lines.append(f"**Hit Prob:** {fmt_prob(hit_prob)} ({fmt_prob(ci_lower)}-{fmt_prob(ci_upper)})")
                    else:
                        desc_lines.append(f"**Hit Prob:** {fmt_prob(hit_prob)}")
                if "hit_streak" in row:
                    desc_lines.append(f"**Hit Streak:** {row.get('hit_streak')}")
                if "hr_prob" in row:
                    hr_prob = row.get('hr_prob')
                    ci_lower = row.get('hr_prob_ci_lower')
                    ci_upper = row.get('hr_prob_ci_upper')
                    if ci_lower and ci_upper:
                        desc_lines.append(f"**HR Prob:** {fmt_prob(hr_prob)} ({fmt_prob(ci_lower)}-{fmt_prob(ci_upper)})")
                    else:
                        desc_lines.append(f"**HR Prob:** {fmt_prob(hr_prob)}")
                if "hr_streak" in row:
                    desc_lines.append(f"**HR Streak:** {row.get('hr_streak')}")
                if "multi_prob" in row:
                    desc_lines.append(f"**Multi-Hit Prob:** {fmt_prob(row.get('multi_prob'))}")
                if "k_prob" in row:
                    k_prob = row.get('k_prob')
                    ci_lower = row.get('k_prob_ci_lower')
                    ci_upper = row.get('k_prob_ci_upper')
                    if ci_lower and ci_upper:
                        desc_lines.append(f"**K Prob:** {fmt_prob(k_prob)} ({fmt_prob(ci_lower)}-{fmt_prob(ci_upper)})")
                    else:
                        desc_lines.append(f"**K Prob:** {fmt_prob(k_prob)}")
                if "k_streak" in row:
                    desc_lines.append(f"**K Streak:** {row.get('k_streak')}")

            desc = "\n".join(desc_lines) if desc_lines else "No model probabilities available."

            color = team_color(team_name) if team_name else 0x1E90FF
            embed = discord.Embed(title=title, description=desc, color=color)

            if player_id:
                embed.set_thumbnail(url=player_headshot_url(int(player_id)))

            footer_parts = []
            if team_name:
                footer_parts.append(team_name)
            if game_date:
                footer_parts.append(f"Date: {game_date}")
            if opp_pitcher_name:
                footer_parts.append(f"Opp Pitcher: {opp_pitcher_name}")

            if footer_parts:
                embed.set_footer(text=" | ".join(footer_parts))

            labels.append(f"{best} – {game_date}")
            embeds.append(embed)

        view = Paginator(embeds, labels=labels, user_id=interaction.user.id)
        await interaction.followup.send(embed=embeds[0], view=view, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /winprops
# -----------------------------
@tree.command(name="winprops", description="Show today's win probabilities.")
async def winprops_cmd(interaction):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading win probabilities…", ephemeral=True)

    try:
        wp_path = DATA_DIR / "fact_win_probability.parquet"
        if not wp_path.exists():
            await interaction.followup.send("Win probability file not found.", ephemeral=True)
            return

        df = pd.read_parquet(wp_path)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        today_date = date.today()
        today_df = df[df["game_date"] == today_date]

        if today_df.empty:
            await interaction.followup.send("No games found for today.", ephemeral=True)
            return

        embeds = []
        labels = []

        for _, r in today_df.iterrows():
            home = r["home_team_name"]
            away = r["away_team_name"]
            hp = r["home_win_prob"]
            ap = r["away_win_prob"]
            conf = r.get("confidence_label", "Unknown")
            pick = r.get("predicted_winner", "N/A")
            game_date = r.get("game_date")
            
            # Betting recommendations if available
            home_rec = r.get("home_recommendation", "N/A")
            away_rec = r.get("away_recommendation", "N/A")
            home_ev = r.get("home_ev")
            away_ev = r.get("away_ev")
            home_edge = r.get("home_edge")
            away_edge = r.get("away_edge")
            home_kelly = r.get("home_kelly")
            away_kelly = r.get("away_kelly")
            home_odds = r.get("home_odds")
            away_odds = r.get("away_odds")

            if pick == home:
                color = team_color(home)
                logo = team_logo(home)
            elif pick == away:
                color = team_color(away)
                logo = team_logo(away)
            else:
                color = 0x32CD32
                logo = None

            # Build description
            desc_lines = [
                f"**Home Win Prob:** {fmt_prob(hp)}",
                f"**Away Win Prob:** {fmt_prob(ap)}",
                f"**Confidence:** {conf}",
                f"**Pick:** {pick}"
            ]
            
            # Add betting info if available
            if home_rec != "N/A" and home_odds is not None:
                desc_lines.append(f"**Home Odds:** {home_odds}")
                desc_lines.append(f"**Home Rec:** {home_rec}")
                if home_ev is not None:
                    desc_lines.append(f"**Home EV:** ${home_ev:.2f}")
                if home_edge is not None:
                    desc_lines.append(f"**Home Edge:** +{home_edge*100:.1f}%")
                if home_kelly is not None and home_kelly > 0:
                    desc_lines.append(f"**Home Kelly:** {home_kelly*100:.1f}%")
            
            if away_rec != "N/A" and away_odds is not None:
                desc_lines.append(f"**Away Odds:** {away_odds}")
                desc_lines.append(f"**Away Rec:** {away_rec}")
                if away_ev is not None:
                    desc_lines.append(f"**Away EV:** ${away_ev:.2f}")
                if away_edge is not None:
                    desc_lines.append(f"**Away Edge:** +{away_edge*100:.1f}%")
                if away_kelly is not None and away_kelly > 0:
                    desc_lines.append(f"**Away Kelly:** {away_kelly*100:.1f}%")
            
            embed = discord.Embed(
                title=f"{away} @ {home}",
                description="\n".join(desc_lines),
                color=color,
            )

            if logo:
                embed.set_thumbnail(url=logo)

            embed.set_footer(text=f"Date: {game_date}")

            labels.append(f"{away} @ {home}")
            embeds.append(embed)

        view = Paginator(embeds, labels=labels, user_id=interaction.user.id)
        await interaction.followup.send(embed=embeds[0], view=view, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /performance
# -----------------------------
@tree.command(name="performance", description="Show model performance metrics.")
async def performance_cmd(interaction):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading performance metrics…", ephemeral=True)

    try:
        perf_data = get_historical_performance(days=7)
        
        if perf_data is None or perf_data.empty:
            await interaction.followup.send("No performance data available yet.", ephemeral=True)
            return

        # Calculate average metrics
        metrics_summary = {}
        for prop in ["hit", "hr", "k"]:
            prop_metrics = []
            for _, row in perf_data.iterrows():
                if prop in row.get("metrics", {}):
                    prop_metrics.append(row["metrics"][prop])
            
            if prop_metrics:
                avg_brier = np.mean([m.get("brier_score", 0) for m in prop_metrics])
                avg_cal = np.mean([m.get("calibration_error", 0) for m in prop_metrics])
                metrics_summary[prop] = {
                    "avg_brier": avg_brier,
                    "avg_calibration": avg_cal
                }

        embed = discord.Embed(
            title="📊 Model Performance (Last 7 Days)",
            description="Brier Score: Lower is better (0-1 scale)\nCalibration Error: Lower is better",
            color=0x1E90FF
        )

        for prop, metrics in metrics_summary.items():
            embed.add_field(
                name=f"{prop.upper()} Model",
                value=f"Brier: {metrics['avg_brier']:.4f}\nCalibration: {metrics['avg_calibration']:.4f}",
                inline=True
            )

        await interaction.followup.send(embed=embed, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /player_history
# -----------------------------
@tree.command(name="player_history", description="Show historical performance for a player.")
@app_commands.describe(player="Player name to search for", days="Number of days to look back (default: 30)")
async def player_history_cmd(interaction, player: str, days: int = 30):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading player history…", ephemeral=True)

    try:
        stats_path = DATA_DIR / "fact_player_stats.parquet"
        game_logs_path = DATA_DIR / "fact_batter_game_logs.parquet"

        if not (stats_path.exists() and game_logs_path.exists()):
            await interaction.followup.send("Missing stats/game_logs files.", ephemeral=True)
            return

        stats_df = pd.read_parquet(stats_path)
        game_logs_df = pd.read_parquet(game_logs_path)

        # Build name list for fuzzy match
        names = stats_df["player_name"].dropna().unique().tolist()
        best = best_player_match(player, names)

        if not best:
            await interaction.followup.send(f"No close match found for '{player}'.", ephemeral=True)
            return

        # Get player ID
        player_data = stats_df[stats_df["player_name"] == best]
        if player_data.empty:
            await interaction.followup.send(f"No stats found for {best}.", ephemeral=True)
            return

        player_id = player_data.iloc[0]["player_id"]

        # Get recent game logs
        player_logs = game_logs_df[game_logs_df["batter_id"] == player_id].copy()
        
        # Filter by date if game_date column exists
        if "game_pk" in player_logs.columns:
            # Get game dates from games table
            games_path = DATA_DIR / "fact_games.parquet"
            if games_path.exists():
                games_df = pd.read_parquet(games_path)
                games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce")
                
                # Merge to get game dates
                player_logs = player_logs.merge(games_df[["game_pk", "game_date"]], on="game_pk", how="left")
                
                # Filter by date range
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                player_logs = player_logs[player_logs["game_date"] >= cutoff_date]

        if player_logs.empty:
            await interaction.followup.send(f"No game logs found for {best} in the last {days} days.", ephemeral=True)
            return

        # Calculate statistics
        total_games = len(player_logs["game_pk"].unique())
        total_ab = player_logs["is_ab"].sum() if "is_ab" in player_logs.columns else 0
        total_hits = player_logs["is_hit"].sum() if "is_hit" in player_logs.columns else 0
        total_hrs = player_logs["is_hr"].sum() if "is_hr" in player_logs.columns else 0
        total_multi = player_logs["is_multi"].sum() if "is_multi" in player_logs.columns else 0
        total_so = player_logs["is_so"].sum() if "is_so" in player_logs.columns else 0
        
        hit_rate = total_hits / total_ab if total_ab > 0 else 0
        hr_rate = total_hrs / total_ab if total_ab > 0 else 0
        multi_rate = total_multi / total_games if total_games > 0 else 0
        k_rate = total_so / total_ab if total_ab > 0 else 0

        # Build embed
        embed = discord.Embed(
            title=f"📈 {best} - Historical Performance (Last {days} Days)",
            description=f"Games Played: {total_games}",
            color=0x1E90FF
        )

        embed.add_field(name="Hit Rate", value=f"{hit_rate:.1%}", inline=True)
        embed.add_field(name="HR Rate", value=f"{hr_rate:.1%}", inline=True)
        embed.add_field(name="Multi-Hit Rate", value=f"{multi_rate:.1%}", inline=True)
        embed.add_field(name="K Rate", value=f"{k_rate:.1%}", inline=True)

        # Add player headshot if available
        embed.set_thumbnail(url=player_headshot_url(int(player_id)))

        await interaction.followup.send(embed=embed, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /matchup
# -----------------------------
@tree.command(name="matchup", description="Analyze batter-pitcher matchup history.")
@app_commands.describe(batter="Batter name", pitcher="Pitcher name", days="Number of days to look back (default: 365)")
async def matchup_cmd(interaction, batter: str, pitcher: str, days: int = 365):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading matchup analysis…", ephemeral=True)

    try:
        stats_path = DATA_DIR / "fact_player_stats.parquet"
        matchups_path = DATA_DIR / "fact_batter_pitcher_matchups.parquet"

        if not (stats_path.exists() and matchups_path.exists()):
            await interaction.followup.send("Missing stats/matchups files.", ephemeral=True)
            return

        stats_df = pd.read_parquet(stats_path)
        matchups_df = pd.read_parquet(matchups_path)

        # Build name lists for fuzzy match
        names = stats_df["player_name"].dropna().unique().tolist()
        best_batter = best_player_match(batter, names)
        best_pitcher = best_player_match(pitcher, names)

        if not best_batter or not best_pitcher:
            await interaction.followup.send(f"No close match found for one or both players.", ephemeral=True)
            return

        # Get player IDs
        batter_data = stats_df[stats_df["player_name"] == best_batter]
        pitcher_data = stats_df[stats_df["player_name"] == best_pitcher]

        if batter_data.empty or pitcher_data.empty:
            await interaction.followup.send(f"Stats not found for one or both players.", ephemeral=True)
            return

        batter_id = batter_data.iloc[0]["player_id"]
        pitcher_id = pitcher_data.iloc[0]["player_id"]

        # Get matchup data
        matchup_data = matchups_df[
            (matchups_df["batter_id"] == batter_id) & 
            (matchups_df["pitcher_id"] == pitcher_id)
        ].copy()

        if matchup_data.empty:
            await interaction.followup.send(f"No matchup history found between {best_batter} and {best_pitcher}.", ephemeral=True)
            return

        # Filter by date if available
        if "game_date" in matchup_data.columns:
            matchup_data["game_date"] = pd.to_datetime(matchup_data["game_date"], errors="coerce")
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            matchup_data = matchup_data[matchup_data["game_date"] >= cutoff_date]

        if matchup_data.empty:
            await interaction.followup.send(f"No matchup history found in the last {days} days.", ephemeral=True)
            return

        # Calculate matchup statistics
        total_pas = matchup_data["is_pa"].sum() if "is_pa" in matchup_data.columns else len(matchup_data)
        hits = matchup_data["is_hit"].sum() if "is_hit" in matchup_data.columns else 0
        hrs = matchup_data["is_hr"].sum() if "is_hr" in matchup_data.columns else 0
        multi_hits = matchup_data["is_multi"].sum() if "is_multi" in matchup_data.columns else 0
        strikeouts = matchup_data["is_so"].sum() if "is_so" in matchup_data.columns else 0

        hit_rate = hits / total_pas if total_pas > 0 else 0
        hr_rate = hrs / total_pas if total_pas > 0 else 0
        multi_rate = multi_hits / total_pas if total_pas > 0 else 0
        k_rate = strikeouts / total_pas if total_pas > 0 else 0

        # Build embed
        embed = discord.Embed(
            title=f"⚔️ Matchup Analysis: {best_batter} vs {best_pitcher}",
            description=f"Last {days} days • {total_pas} plate appearances",
            color=0x1E90FF
        )

        embed.add_field(name="Hit Rate", value=f"{hit_rate:.1%}", inline=True)
        embed.add_field(name="HR Rate", value=f"{hr_rate:.1%}", inline=True)
        embed.add_field(name="Multi-Hit Rate", value=f"{multi_rate:.1%}", inline=True)
        embed.add_field(name="K Rate", value=f"{k_rate:.1%}", inline=True)

        # Add player headshots
        embed.set_thumbnail(url=player_headshot_url(int(batter_id)))
        embed.set_footer(text=f"Pitcher ID: {pitcher_id}")

        await interaction.followup.send(embed=embed, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /top_picks
# -----------------------------
@tree.command(name="top_picks", description="Show today's highest probability predictions.")
async def top_picks_cmd(interaction):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading top picks…", ephemeral=True)

    try:
        props_path = DATA_DIR / "fact_player_props.parquet"
        if not props_path.exists():
            await interaction.followup.send("Player props file not found.", ephemeral=True)
            return

        df = pd.read_parquet(props_path)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        today_date = date.today()
        df = df[df["game_date"] == today_date]

        if df.empty:
            await interaction.followup.send("No props found for today.", ephemeral=True)
            return

        embeds = []
        
        # Top hit predictions
        top_hit = df.nlargest(5, "hit_prob")[["player_name", "team_name", "hit_prob", "hit_streak"]]
        hit_desc = "\n".join([
            f"{row['player_name']} ({row['team_name']}) - {fmt_prob(row['hit_prob'])} {row['hit_streak']}"
            for _, row in top_hit.iterrows()
        ])
        
        embed = discord.Embed(
            title="🔥 Top Hit Predictions",
            description=hit_desc,
            color=0x1E90FF
        )
        embeds.append(embed)

        # Top HR predictions
        top_hr = df.nlargest(5, "hr_prob")[["player_name", "team_name", "hr_prob", "hr_streak"]]
        hr_desc = "\n".join([
            f"{row['player_name']} ({row['team_name']}) - {fmt_prob(row['hr_prob'])} {row['hr_streak']}"
            for _, row in top_hr.iterrows()
        ])
        
        embed = discord.Embed(
            title="🏆 Top HR Predictions",
            description=hr_desc,
            color=0xFFD700
        )
        embeds.append(embed)

        # Top multi-hit predictions
        top_multi = df.nlargest(5, "multi_prob")[["player_name", "team_name", "multi_prob"]]
        multi_desc = "\n".join([
            f"{row['player_name']} ({row['team_name']}) - {fmt_prob(row['multi_prob'])}"
            for _, row in top_multi.iterrows()
        ])
        
        embed = discord.Embed(
            title="💥 Top Multi-Hit Predictions",
            description=multi_desc,
            color=0xFF6347
        )
        embeds.append(embed)

        view = Paginator(embeds, labels=["Top Hits", "Top HRs", "Top Multi-Hits"], user_id=interaction.user.id)
        await interaction.followup.send(embed=embeds[0], view=view, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /hot_players
# -----------------------------
@tree.command(name="hot_players", description="Show players with hot streaks.")
async def hot_players_cmd(interaction):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading hot players…", ephemeral=True)

    try:
        props_path = DATA_DIR / "fact_player_props.parquet"
        if not props_path.exists():
            await interaction.followup.send("Player props file not found.", ephemeral=True)
            return

        df = pd.read_parquet(props_path)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        today_date = date.today()
        df = df[df["game_date"] == today_date]

        if df.empty:
            await interaction.followup.send("No props found for today.", ephemeral=True)
            return

        # Filter for hot players with meaningful probabilities (limit to 10 to avoid character limit)
        hot_hit = df[(df["hit_streak"] == "🔥 HOT") & (df["hit_prob"] > 0.20)][["player_name", "team_name", "hit_prob"]].head(10)
        hot_hr = df[(df["hr_streak"] == "🔥 HOT") & (df["hr_prob"] > 0.05)][["player_name", "team_name", "hr_prob"]].head(10)
        hot_k = df[(df["k_streak"] == "🔥 HOT") & (df["k_prob"] > 0.20)][["player_name", "team_name", "k_prob"]].head(10)

        embeds = []

        if not hot_hit.empty:
            hit_desc = "\n".join([
                f"{row['player_name']} ({row['team_name']}) - {fmt_prob(row['hit_prob'])}"
                for _, row in hot_hit.iterrows()
            ])
            embed = discord.Embed(
                title="🔥 Hot Hitters",
                description=hit_desc,
                color=0xFF4500
            )
            embeds.append(embed)
        else:
            embed = discord.Embed(
                title="🔥 Hot Hitters",
                description="No hot hitters today",
                color=0xFF4500
            )
            embeds.append(embed)

        if not hot_hr.empty:
            hr_desc = "\n".join([
                f"{row['player_name']} ({row['team_name']}) - {fmt_prob(row['hr_prob'])}"
                for _, row in hot_hr.iterrows()
            ])
            embed = discord.Embed(
                title="🏆 Hot HR Hitters",
                description=hr_desc,
                color=0xFFD700
            )
            embeds.append(embed)
        else:
            embed = discord.Embed(
                title="🏆 Hot HR Hitters",
                description="No hot HR hitters today",
                color=0xFFD700
            )
            embeds.append(embed)

        if not hot_k.empty:
            k_desc = "\n".join([
                f"{row['player_name']} ({row['team_name']}) - {fmt_prob(row['k_prob'])}"
                for _, row in hot_k.iterrows()
            ])
            embed = discord.Embed(
                title="✅ Hot Contact (Low K)",
                description=k_desc,
                color=0x32CD32
            )
            embeds.append(embed)
        else:
            embed = discord.Embed(
                title="✅ Hot Contact (Low K)",
                description="No hot contact players today",
                color=0x32CD32
            )
            embeds.append(embed)

        view = Paginator(embeds, labels=["Hot Hitters", "Hot HR", "Hot Contact"], user_id=interaction.user.id)
        await interaction.followup.send(embed=embeds[0], view=view, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /props (team props)
# -----------------------------
@tree.command(name="props", description="Show today's team props.")
async def props_cmd(interaction):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading team props…", ephemeral=True)

    try:
        tp_path = DATA_DIR / "fact_team_props.parquet"
        if not tp_path.exists():
            await interaction.followup.send("Team props file not found.", ephemeral=True)
            return

        df = pd.read_parquet(tp_path)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        today_date = date.today()
        today_df = df[df["game_date"] == today_date]

        if today_df.empty:
            await interaction.followup.send("No games found for today.", ephemeral=True)
            return

        embeds = []
        labels = []

        for _, r in today_df.iterrows():
            home = r["home_team_name"]
            away = r["away_team_name"]

            h_line = r["home_team_total_runs_line"]
            a_line = r["away_team_total_runs_line"]
            g_line = r["game_total_runs_line"]
            h_ml = r["home_team_moneyline"]
            a_ml = r["away_team_moneyline"]
            h_wp = r["home_win_prob"]
            a_wp = r["away_win_prob"]
            h_conf = r["home_confidence"]
            a_conf = r["away_confidence"]
            game_date = r["game_date"]
            pred = r.get("predicted_winner", home)

            color = team_color(pred)
            logo = team_logo(pred)

            embed = discord.Embed(
                title=f"{away} @ {home}",
                description=(
                    f"**Home Team Total Runs Line:** {h_line}\n"
                    f"**Away Team Total Runs Line:** {a_line}\n"
                    f"**Game Total Runs Line:** {g_line}\n"
                    f"**Home Moneyline:** {h_ml}\n"
                    f"**Away Moneyline:** {a_ml}\n"
                    f"**Home Win Prob:** {fmt_prob(h_wp)}\n"
                    f"**Away Win Prob:** {fmt_prob(a_wp)}\n"
                    f"**Home Confidence:** {h_conf}\n"
                    f"**Away Confidence:** {a_conf}\n"
                    f"**Predicted Winner:** {pred}"
                ),
                color=color,
            )

            if logo:
                embed.set_thumbnail(url=logo)

            embed.set_footer(text=f"Date: {game_date}")

            labels.append(f"{away} @ {home}")
            embeds.append(embed)

        view = Paginator(embeds, labels=labels, user_id=interaction.user.id)
        await interaction.followup.send(embed=embeds[0], view=view, ephemeral=True)

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# COMMAND: /matchups (POWER MODE)
# -----------------------------

@tree.command(name="matchups", description="Show today's batter vs pitcher matchups by game.")
async def matchups_cmd(interaction):
    if interaction.user.id != USER_ID:
        return

    await interaction.response.send_message("Loading matchups…", ephemeral=True)

    try:
        matchups_path = DATA_DIR / "fact_batter_pitcher_matchups.parquet"
        games_path = DATA_DIR / "fact_games.parquet"
        wp_path = DATA_DIR / "fact_win_probability.parquet"

        mdf = pd.read_parquet(matchups_path)
        gdf = pd.read_parquet(games_path)
        wp_df = pd.read_parquet(wp_path)

        # FIX: define today_date
        today_date = date.today()

        # Convert game_date
        gdf["game_date"] = pd.to_datetime(gdf["game_date"], errors="coerce").dt.date

        # Filter games for today
        today_games = gdf[gdf["game_date"] == today_date]

        if today_games.empty:
            await interaction.followup.send("No games found for today.", ephemeral=True)
            return

        # Filter matchups to ONLY today's games
        mdf = mdf[mdf["game_pk"].isin(today_games["game_pk"])]

        # If no historical matchups for today (scheduled games), use fact_player_props.parquet
        if mdf.empty:
            props_path = DATA_DIR / "fact_player_props.parquet"
            stats_path = DATA_DIR / "fact_player_stats.parquet"
            if not props_path.exists():
                await interaction.followup.send("Player props not scored yet. Run /run first.", ephemeral=True)
                return
            
            props_df = pd.read_parquet(props_path)
            props_df["game_date"] = pd.to_datetime(props_df["game_date"], errors="coerce").dt.date
            props_today = props_df[props_df["game_date"] == today_date]
            
            if props_today.empty:
                await interaction.followup.send("No props found for today.", ephemeral=True)
                return
            
            # Get pitcher names from player_stats
            if stats_path.exists():
                stats_df = pd.read_parquet(stats_path)
                pitcher_names = stats_df[stats_df["group"] == "pitching"][["player_id", "player_name"]].drop_duplicates("player_id")
                pitcher_names = pitcher_names.rename(columns={"player_id": "pitcher_id", "player_name": "pitcher_name"})
                
                # Get batter team IDs from player_stats
                batter_teams = stats_df[stats_df["group"] == "hitting"][["player_id", "team_id"]].drop_duplicates("player_id")
                batter_teams = batter_teams.rename(columns={"player_id": "batter_id"})
            else:
                pitcher_names = pd.DataFrame(columns=["pitcher_id", "pitcher_name"])
                batter_teams = pd.DataFrame(columns=["batter_id", "team_id"])
            
            # Use props as the matchup data for scheduled games
            mdf = props_today[["player_id", "player_name", "game_pk", "opp_pitcher_id", "hit_prob", "hr_prob", "multi_prob", "k_prob", "home_team_name", "away_team_name", "game_date"]].copy()
            mdf = mdf.rename(columns={"player_id": "batter_id", "player_name": "batter_name", "opp_pitcher_id": "pitcher_id"})
            
            # Merge batter team IDs
            mdf = mdf.merge(batter_teams, on="batter_id", how="left")
            
            # Merge team IDs from games file
            mdf = mdf.merge(
                today_games[["game_pk", "home_team_id", "away_team_id"]],
                on="game_pk",
                how="left"
            )
            
            # Merge pitcher names
            mdf = mdf.merge(pitcher_names, on="pitcher_id", how="left")
        else:
            # Merge in team names + date for historical matchups
            mdf = mdf.merge(
                today_games[["game_pk", "home_team_name", "away_team_name", "game_date", "home_team_id", "away_team_id"]],
                on="game_pk",
                how="left",
            )
            
            mdf["game_date"] = pd.to_datetime(mdf["game_date"], errors="coerce").dt.date

            # Merge batter team IDs from player stats so home/away can be split properly
            stats_path = DATA_DIR / "fact_player_stats.parquet"
            if stats_path.exists():
                stats_df = pd.read_parquet(stats_path)
                batter_teams = stats_df[stats_df["group"] == "hitting"][["player_id", "team_id"]].drop_duplicates("player_id")
                batter_teams = batter_teams.rename(columns={"player_id": "batter_id"})
                mdf = mdf.merge(batter_teams, on="batter_id", how="left")

            # Merge in ML probability columns from score_batter_props output
            props_path = DATA_DIR / "fact_player_props.parquet"
            if props_path.exists():
                props_df = pd.read_parquet(props_path)
                props_df["game_date"] = pd.to_datetime(props_df["game_date"], errors="coerce").dt.date
                props_today = props_df[props_df["game_date"] == today_date]
                mdf = mdf.merge(
                    props_today[["player_id", "game_pk", "hit_prob", "hr_prob", "multi_prob", "k_prob"]],
                    left_on=["batter_id", "game_pk"],
                    right_on=["player_id", "game_pk"],
                    how="left",
                )
            else:
                print("⚠️  fact_player_props.parquet not found — run score_batter_props first.")
                await interaction.followup.send("Player props not scored yet. Run /run first.", ephemeral=True)
                return

        # Filter winprob for today
        wp_df["game_date"] = pd.to_datetime(wp_df["game_date"], errors="coerce").dt.date
        wp_today = wp_df[wp_df["game_date"] == today_date].set_index("game_pk")

        # Build game list (sorted by team names for easier reading)
        games = sorted({
            (int(r["game_pk"]), r["home_team_name"], r["away_team_name"])
            for _, r in mdf.iterrows()
        }, key=lambda x: (x[1], x[2]))  # Sort by home_team_name, then away_team_name

        # Build embeds
        embeds, labels = build_top20_embeds(mdf, discord)

        if not embeds:
            await interaction.followup.send("No Top 20 matchups available.", ephemeral=True)
            return

        paginator = Paginator(embeds, labels=labels, user_id=interaction.user.id)

        paginator.add_item(
            JumpToGamesButton(
                mdf,
                wp_today,
                games,
                interaction.user.id,
                paginator_cls=Paginator,
                team_color_fn=team_color,
                team_logo_fn=team_logo
            )
        )

        await interaction.followup.send(
            content="Top 20 Danger‑Score Matchups",
            embed=embeds[0],
            view=paginator,
            ephemeral=True,
        )

    except Exception as e:
        traceback.print_exc()
        await interaction.followup.send(f"Error: {e}", ephemeral=True)


# -----------------------------
# RUN BOT
# -----------------------------
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Set DISCORD_BOT_TOKEN environment variable
if not TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
client.run(TOKEN)
