#!/usr/bin/env python3

import pytest
import time
import filecmp

from render import *


@pytest.fixture
def events_off_first():
    return load_events_files(["test_data/test_off_first.txt"])


@pytest.fixture
def events_on_first():
    return load_events_files(["test_data/test_on_first.txt"])


@pytest.fixture
def segments_off_first(events_off_first):
    return separate_on_off_segments(make_segments(events_off_first))


@pytest.fixture
def segments_on_first(events_on_first):
    return separate_on_off_segments(make_segments(events_on_first))


@pytest.fixture
def trend_segments_with_values(events_on_first):
    return calc_trend_segment_values(split_segments(make_trend_segments(events_on_first, seconds_per_day)))


@pytest.fixture
def cohesion_segments_with_values(events_on_first):
    trend_segments = make_trend_segments(events_on_first, seconds_per_day)
    cohesion_segments = split_segments(make_cohesion_segments(trend_segments, seconds_per_day))
    cohesion_segments = calc_trend_segment_values(cohesion_segments, seconds_per_day)
    return cohesion_segments


@pytest.fixture
def trend_lut():
    return colormap_to_lut(trend_colormap)


@pytest.fixture
def cohesion_lut():
    return colormap_to_lut(cohesion_colormap)


def test_load_events_files(events_off_first, events_on_first):
    assert len(events_on_first) == 11
    assert events_on_first[0].state == OnOff.on
    assert events_on_first[0].time.hour == 0
    assert events_on_first[1].state == OnOff.off
    assert events_on_first[1].time.hour == 1
    assert events_on_first[2].state == OnOff.on
    assert events_on_first[2].time.hour == 3
    assert events_on_first[10].state == OnOff.on
    assert events_on_first[10].time.hour == 10

    assert len(events_off_first) == 10
    assert events_off_first[0].state == OnOff.off
    assert events_off_first[0].time.hour == 1
    assert events_off_first[1].state == OnOff.on
    assert events_off_first[1].time.hour == 3
    assert events_off_first[9].state == OnOff.on
    assert events_off_first[9].time.hour == 10


def test_make_segments_off_first(segments_off_first):
    off_segments, on_segments = segments_off_first
    
    assert len(off_segments) == 5
    assert off_segments[0].start.hour == 1
    assert off_segments[1].start.hour == 6
    assert off_segments[2].start.hour == 2
    assert off_segments[3].start.hour == 9
    assert off_segments[4].start.hour == 7

    assert len(on_segments) == 4
    assert on_segments[0].start.hour == 3
    assert on_segments[1].start.hour == 9
    assert on_segments[2].start.hour == 5
    assert on_segments[3].start.hour == 3


def test_make_segments_on_first(segments_on_first):
    off_segments, on_segments = segments_on_first
    
    assert len(off_segments) == 5
    assert off_segments[0].start.hour == 1
    assert off_segments[1].start.hour == 6
    assert off_segments[2].start.hour == 2
    assert off_segments[3].start.hour == 9
    assert off_segments[4].start.hour == 7

    assert len(on_segments) == 5
    assert on_segments[0].start.hour == 0
    assert on_segments[1].start.hour == 3
    assert on_segments[2].start.hour == 9
    assert on_segments[3].start.hour == 5
    assert on_segments[4].start.hour == 3


def test_events_from_segments(events_on_first):
    segments = make_segments(events_on_first)
    events = events_from_segments(segments)
    assert events_on_first == events


def test_split_segments(segments_on_first):
    off_segments, on_segments = segments_on_first

    split_off_segments = split_segments(off_segments)
    assert len(split_off_segments) == 6
    assert split_off_segments[0].end.hour == 3
    assert split_off_segments[3].end.hour == 23
    assert split_off_segments[4].start.hour == 0
    assert split_off_segments[4].start.minute == 0
    assert split_off_segments[4].start.second == 0
    assert split_off_segments[4].end.hour == 3

    split_on_segments  = split_segments(on_segments)
    assert len(split_on_segments) == 6
    assert split_on_segments[0].end.hour == 1
    assert split_on_segments[2].end.hour == 23
    assert split_on_segments[3].start.hour == 0
    assert split_on_segments[3].start.minute == 0
    assert split_on_segments[3].start.second == 0


def test_make_trend_segments(events_off_first, events_on_first):
    trend_segments = make_trend_segments(events_off_first, seconds_per_day)
    assert len(trend_segments) == 15  # two less because the tail doesn't hit the 'Mar 21 00:00:00' breakpoint to divide the 9 to 1 segment 
    assert [s.end.hour for s in trend_segments] == [     3,  6,  9,      1,  2,  3,  5,  6,  9,  2,  3,  5,  7,  9, 10]
    assert [s.end.day for s in trend_segments] == [    21, 21, 21,     22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23]
    assert [s.trend for s in trend_segments]       == [     1,  0,  1,      0, -1,  0,  1,  0, -1,  1,  0, -1,  0,  1,  0]

    trend_segments = make_trend_segments(events_on_first, seconds_per_day)
    assert len(trend_segments) == 17
    assert [s.end.hour for s in trend_segments] == [ 1,  3,  6,  9,  0,  1,  2,  3,  5,  6,  9,  2,  3,  5,  7,  9, 10]
    assert [s.end.day for s in trend_segments] == [21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23]
    assert [s.trend for s in trend_segments]       == [ 0,  1,  0,  1,  0,  0, -1,  0,  1,  0, -1,  1,  0, -1,  0,  1,  0]


def test_calc_trend_segment_values(trend_segments_with_values):
    assert len(trend_segments_with_values) == 19  # beware an extra zero-length segment when dealing with timestamps at exactly midnight, due to 23:59:60 and 00:00:00 being different
    assert [s.end_value/(60*60) for s in trend_segments_with_values] == [0, 2, 2, 5, 5, 5, 5, 4, 4, 6, 6, 3, 18, 20, 20, 18, 18, 20, 20]


def test_make_cohesion_segments(events_on_first):
    trend_segments = make_trend_segments(events_on_first, seconds_per_day)
    cohesion_segments = make_cohesion_segments(trend_segments, seconds_per_day)
    assert len(cohesion_segments) == 20
    assert [s.end.hour for s in cohesion_segments] == [ 1,  3,  6,  9,  0,  1,  2,  3,  5,  6,  9,  0,  1,  2,  3,  5,  6,  7,  9, 10]
    assert [s.end.day for s in cohesion_segments] == [21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23]
    assert [s.trend for s in cohesion_segments]       == [ 0, -1,  0, -1,  0,  0,  0,  1, -1,  0,  0, -1, -1,  0,  0,  0,  0,  1,  0,  1]


def test_calc_trend_segment_values__cohesion(cohesion_segments_with_values):
    assert len(cohesion_segments_with_values) == 22  # two extra segments because of the 23:59:60 -> 00:00:00 thing
    assert [s.end_value/(60*60) for s in cohesion_segments_with_values] == [24, 22, 22, 19, 19, 19, 19, 19, 20, 18, 18, 18, 3, 3, 2, 2, 2, 2, 2, 3, 3, 4]


@pytest.mark.skip
def test_calc_trend_segment_values_with_dst():
    pass  #todo


def test_draw_segment_chart_and_frame(segments_on_first, tmp_path):
    off_segments, on_segments = segments_on_first
    split_off_segments = split_segments(off_segments)
    
    chart_params = draw_segment_chart(split_off_segments, width=720, day_height=6)
    chart_image, day_height, day_count, start_time, lut, key_min, key_max = chart_params
    
    assert day_height == 6
    assert day_count == 3
    assert start_time == off_segments[0].start
    assert lut == []
    assert key_min is None
    assert key_max is None

    fname = tmp_path / "test_draw_segment_chart_and_frame.png"
    draw_chart_frame(*chart_params).save(fname)

    assert filecmp.cmp(fname, "test_data/correct_output/history.png", False)


def test_draw_trend_chart_and_frame(trend_segments_with_values, trend_lut, tmp_path):
    chart_params = draw_trend_chart(trend_segments_with_values, 1 * seconds_per_day, trend_lut, normalized=True)
    chart_image, day_height, day_count, start_time, lut, key_min, key_max = chart_params

    assert day_height == 6
    assert day_count == 3
    assert start_time == trend_segments_with_values[0].start
    assert lut == trend_lut
    assert key_min == 3/24
    assert key_max == 20/24

    fname = tmp_path / "test_draw_trend_chart_and_frame.png"
    draw_chart_frame(*chart_params, gridcolor=grid_line_color).save(fname)

    assert filecmp.cmp(fname, "test_data/correct_output/trend_daily.png", False)


def test_draw_cohesion_chart_and_frame(cohesion_segments_with_values, cohesion_lut, tmp_path):
    chart_params = draw_trend_chart(cohesion_segments_with_values, 1 * seconds_per_day, cohesion_lut, normalized=True, extra_ignore_interval=seconds_per_day)
    chart_image, day_height, day_count, start_time, lut, key_min, key_max = chart_params

    assert day_height == 6
    assert day_count == 3
    assert start_time == cohesion_segments_with_values[0].start
    assert lut == cohesion_lut
    assert key_min == 2/24
    assert key_max == 4/24

    fname = tmp_path / "test_draw_cohesion_chart_and_frame.png"
    draw_chart_frame(*chart_params, gridcolor=grid_line_color, percentage=True).save(fname)

    assert filecmp.cmp(fname, "test_data/correct_output/cohesion_daily.png", False)


def test_draw_plots(segments_on_first, events_on_first, tmp_path):
    off_segments, on_segments = segments_on_first
    
    trend_segments = {}
    cohesion_segments = {}
    for days,name in intervals.items():
        interval = days * seconds_per_day
        trend_segments[days] = calc_trend_segment_values(split_segments(make_trend_segments(events_on_first, interval)))
        cohesion_segments[days] = calc_trend_segment_values(split_segments(make_cohesion_segments(trend_segments[1], interval)), interval)

    figs = draw_plots(off_segments, on_segments, trend_segments, cohesion_segments)
    
    for i,fig in enumerate(figs):
        fname = tmp_path / f"plots{i+1}.png"
        fig.savefig(fname)
        
        assert filecmp.cmp(fname, f"test_data/correct_output/plots{i+1}.png", False)


def test_segment_length_hours(segments_on_first):
    off_segments, on_segments = segments_on_first
    off_segment_lengths = segment_length_hours(off_segments)
    on_segment_lengths = segment_length_hours(on_segments)
    assert off_segment_lengths == [2, 3, 3, 18, 3]
    assert on_segment_lengths  == [1, 3, 17, 4, 4]


def test_make_bins():
    assert make_bins([13.1], 0.5) == [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5]
    assert make_bins([24.2], 2) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    assert all(acc - exp < 1e-8 for acc,exp in zip(make_bins([2], 0.2), [0, .2, .4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2]))


def test_format_hours():
    assert format_hours(2.5)   == " 2:30"
    assert format_hours(2.25)  == " 2:15"
    assert format_hours(2.00)  == " 2:00"
    assert format_hours(0)     == " 0:00"
    assert format_hours(10)    == "10:00"
    assert format_hours(10.99) == "10:59"
    assert format_hours(10.999) == "11:00"
    assert format_hours(22 + 37/60) == "22:37"
    assert format_hours(18.6852) == "18:41"
    assert format_hours(1.001) == " 1:00"


def test_start_of_day():
    times = (
        ("Mon Jun 26 22:31:40 1950", "Mon Jun 26 00:00:00 1950"),
        ("Tue Jun 27 03:15:51 1950", "Tue Jun 27 00:00:00 1950"),
        ("Thu Jun 29 00:00:00 1950", "Thu Jun 29 00:00:00 1950"),
        ("Fri Jun 30 23:59:59 1950", "Fri Jun 30 00:00:00 1950"),
    )
    for t in times:
        assert start_of_day(datetime.strptime(t[0], TIME_FORMAT)).strftime(TIME_FORMAT) == t[1]


def test_end_of_day():
    times = (
        ("Mon Jun 26 22:31:40 1950", "1950-06-26 23:59:59.999999"),
        ("Tue Jun 27 03:15:51 1950", "1950-06-27 23:59:59.999999"),
        ("Thu Jun 29 00:00:00 1950", "1950-06-29 23:59:59.999999"),
        ("Fri Jun 30 23:59:59 1950", "1950-06-30 23:59:59.999999"),
    )
    for t in times:
        assert str(end_of_day(datetime.strptime(t[0], TIME_FORMAT))) == t[1]
