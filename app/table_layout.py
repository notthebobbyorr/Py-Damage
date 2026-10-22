"""Consistent statistic ordering without changing values or adding hidden columns."""

RAW_COUNT_COLUMNS = {
    '#', 'Pitches', 'Pitches Thrown', 'Pitches Seen', 'PA', 'TBF', 'G', 'GS', 'IP',
    'BBE', 'BBE Allowed', 'BBE Against', 'HR', 'H', 'HA', 'XBH', 'BB', 'K',
    'Swings', 'Whiffs', 'Chases', 'Damage BBE', 'Pulled FBs', 'Strikes', 'Balls',
    'Zone Pitches', 'Out of Zone', 'FA#', 'BR#', 'OFF#', 'vs RHP', 'vs LHP',
    'vs LHB', 'vs RHB', 'Selective Takes', 'Hittable Takes', 'SB', 'SBO',
    'LA>=20', 'LA >= 20', 'LA<=0', 'Zone',
}
IDENTITY_COLUMNS = {
    'Name', 'Player ID', 'Team', 'Season', 'Level', 'Level ID', 'Pitch Type',
    'Pitch Group', 'Split', 'Split Type', 'Opponent Hand', 'Pitcher Hand', 'Batter Hand',
    'Home / Away', 'Home/Away', 'Half', 'Month', 'Date', 'Game ID', 'vs',
    'Park ID', 'Home Team',
}


def order_stat_columns(columns, pitching=False, damage_after_la=False, pitch_outcomes_after_csw=False):
    ordered = list(columns)
    counts = [c for c in ['#', 'Pitches', 'Pitches Thrown', 'Pitches Seen', 'BBE', 'BBE Allowed', 'BBE Against', 'HR', 'PA', 'TBF', 'G', 'GS', 'IP'] if c in ordered]
    counts += [c for c in ordered if c in RAW_COUNT_COLUMNS and c not in counts]
    identities = [c for c in ordered if c in IDENTITY_COLUMNS]
    ordered = identities + counts + [c for c in ordered if c not in identities and c not in counts]

    def move(columns_to_move, anchors, before=False):
        nonlocal ordered
        moving = [c for c in columns_to_move if c in ordered]
        anchor = next((c for c in anchors if c in ordered and c not in moving), None)
        if not moving or anchor is None:
            return
        ordered = [c for c in ordered if c not in moving]
        index = ordered.index(anchor) + (0 if before else 1)
        ordered[index:index] = moving

    whiff = ['Whiff (%)', 'Whiff%', 'Whiff Pctile']
    swing = ['Swing (%)', 'Swing%', 'Swing', 'Swing Pctile']
    swstr = ['SwStr (%)', 'SwStr%', 'SwStr', 'SwStr Pctile']
    distance = ['Avg Miss Distance (in.)', 'Avg Miss Distance Pctile']
    if '#' in ordered:
        move(['%'], ['BBE'], before=True)
    move(['SBO', 'SB'], ['Takeoff%', 'Takeoff% Against', 'Takeoff Against (%)'], before=True)
    if pitching:
        move(whiff + swing, swstr, before=True)
        move(distance, ['pSwStr (%)', 'pSwStr%', 'pSwStr Pctile'] + swstr)
        move(['Damage/BBE (%)', 'Damage/BBE%'], distance)
        if damage_after_la:
            move(['Damage/BBE (%)', 'Damage/BBE%', 'pDamage/BBE (%)', 'pDamage/BBE%'],
                 ['LA<=0%', 'LA<=0 (%)', 'LA <= 0', 'LA<=0'])
        if pitch_outcomes_after_csw:
            move(['LA<=0%', 'LA<=0 (%)', 'LA <= 0', 'LA<=0',
                  'Damage/BBE (%)', 'Damage/BBE%', 'pDamage/BBE (%)', 'pDamage/BBE%'],
                 ['CSW (%)', 'CSW%', 'CSW'])
    else:
        move(distance, ['Avg Swing Speed', 'Avg Swing Speed Pctile'], before=True)
        move(whiff, ['Whiff vs. Secondaries (%)', 'Whiff vs Secondaries', 'Whiff vs. Secondaries'], before=True)
        move(swing, ['Chase (%)', 'Chase%', 'Chase'], before=True)
        move(swstr, ['Contact Over Expected (%)', 'Contact Over Expected', 'Contact Over Exp'])
    return ordered
