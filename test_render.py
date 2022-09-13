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
    return make_segments(events_off_first)


@pytest.fixture
def segments_on_first(events_on_first):
    return make_segments(events_on_first)


def test_load_events_files(events_off_first, events_on_first):
    assert len(events_on_first) == 11
    assert events_on_first[0].state == OnOff.on
    assert events_on_first[0].time.tm_hour == 0
    assert events_on_first[1].state == OnOff.off
    assert events_on_first[1].time.tm_hour == 1
    assert events_on_first[2].state == OnOff.on
    assert events_on_first[2].time.tm_hour == 3
    assert events_on_first[10].state == OnOff.on
    assert events_on_first[10].time.tm_hour == 10

    assert len(events_off_first) == 10
    assert events_off_first[0].state == OnOff.off
    assert events_off_first[0].time.tm_hour == 1
    assert events_off_first[1].state == OnOff.on
    assert events_off_first[1].time.tm_hour == 3
    assert events_off_first[9].state == OnOff.on
    assert events_off_first[9].time.tm_hour == 10


def test_make_segments_off_first(segments_off_first):
    off_segments, on_segments = segments_off_first
    
    assert len(off_segments) == 5
    assert off_segments[0].start.tm_hour == 1
    assert off_segments[1].start.tm_hour == 6
    assert off_segments[2].start.tm_hour == 2
    assert off_segments[3].start.tm_hour == 9
    assert off_segments[4].start.tm_hour == 7

    assert len(on_segments) == 4
    assert on_segments[0].start.tm_hour == 3
    assert on_segments[1].start.tm_hour == 9
    assert on_segments[2].start.tm_hour == 5
    assert on_segments[3].start.tm_hour == 3


def test_make_segments_on_first(segments_on_first):
    off_segments, on_segments = segments_on_first
    
    assert len(off_segments) == 5
    assert off_segments[0].start.tm_hour == 1
    assert off_segments[1].start.tm_hour == 6
    assert off_segments[2].start.tm_hour == 2
    assert off_segments[3].start.tm_hour == 9
    assert off_segments[4].start.tm_hour == 7

    assert len(on_segments) == 5
    assert on_segments[0].start.tm_hour == 0
    assert on_segments[1].start.tm_hour == 3
    assert on_segments[2].start.tm_hour == 9
    assert on_segments[3].start.tm_hour == 5
    assert on_segments[4].start.tm_hour == 3
    

def test_split_segments(segments_on_first):
    off_segments, on_segments = segments_on_first

    split_off_segments = split_segments(off_segments)
    assert len(split_off_segments) == 6
    assert split_off_segments[0].end.tm_hour == 3
    assert split_off_segments[3].end.tm_hour == 23
    assert split_off_segments[4].start.tm_hour == 0
    assert split_off_segments[4].start.tm_min == 0
    assert split_off_segments[4].start.tm_sec == 0
    assert split_off_segments[4].end.tm_hour == 3

    split_on_segments  = split_segments(on_segments)
    assert len(split_on_segments) == 6
    assert split_on_segments[0].end.tm_hour == 1
    assert split_on_segments[2].end.tm_hour == 23
    assert split_on_segments[3].start.tm_hour == 0
    assert split_on_segments[3].start.tm_min == 0
    assert split_on_segments[3].start.tm_sec == 0


def test_make_trend_segments(events_off_first, events_on_first):
    trend_segments = make_trend_segments(events_off_first, seconds_per_day)
    assert len(trend_segments) == 15  # two less because the tail doesn't hit the 'Mar 21 00:00:00' breakpoint to divide the 9 to 1 segment 
    assert [s.end.tm_hour for s in trend_segments] == [     3,  6,  9,      1,  2,  3,  5,  6,  9,  2,  3,  5,  7,  9, 10]
    assert [s.end.tm_mday for s in trend_segments] == [    21, 21, 21,     22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23]
    assert [s.trend for s in trend_segments]       == [     1,  0,  1,      0, -1,  0,  1,  0, -1,  1,  0, -1,  0,  1,  0]

    trend_segments = make_trend_segments(events_on_first, seconds_per_day)
    assert len(trend_segments) == 17
    assert [s.end.tm_hour for s in trend_segments] == [ 1,  3,  6,  9,  0,  1,  2,  3,  5,  6,  9,  2,  3,  5,  7,  9, 10]
    assert [s.end.tm_mday for s in trend_segments] == [21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23]
    assert [s.trend for s in trend_segments]       == [ 0,  1,  0,  1,  0,  0, -1,  0,  1,  0, -1,  1,  0, -1,  0,  1,  0]


@pytest.mark.skip
def test_make_cohesion_segments():
    pass


def test_calc_trend_segment_values(events_on_first):
    trend_segments = calc_trend_segment_values(split_segments(make_trend_segments(events_on_first, seconds_per_day)))
    for s in trend_segments:
        print(s)
    assert len(trend_segments) == 19  # beware an extra zero-length segment when dealing with timestamps at exactly midnight, due to 23:59:60 and 00:00:00 being different
    assert [s.end_value/(60*60) for s in trend_segments] == [0, 2, 2, 5, 5, 5, 5, 4, 4, 6, 6, 3, 18, 20, 20, 18, 18, 20, 20]


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
    assert lut is None
    assert key_min is None
    assert key_max is None

    fname = tmp_path / "test_draw_segment_chart_and_frame.png"
    draw_chart_frame(*chart_params).save(fname)

    assert filecmp.cmp(fname, "test_data/correct_output/history.png", False)


@pytest.mark.skip
def test_draw_trend_chart_and_frame():
    pass


@pytest.mark.skip
def test_draw_cohesion_chart_and_frame():
    pass


@pytest.mark.skip
def test_draw_plots():
    pass


def test_segment_length_hours(segments_on_first):
    off_segments, on_segments = segments_on_first
    off_segment_lengths = segment_length_hours(off_segments)
    on_segment_lengths = segment_length_hours(on_segments)
    assert off_segment_lengths == [2, 3, 3, 18, 3]
    assert on_segment_lengths  == [1, 3, 17, 4, 4]


def test_format_hours():
    assert format_hours(2.5)   == " 2:30"
    assert format_hours(2.25)  == " 2:15"
    assert format_hours(2.00)  == " 2:00"
    assert format_hours(0)     == " 0:00"
    assert format_hours(10)    == "10:00"
    #assert format_hours(10.99) == "11:00"  #woops
    assert format_hours(22 + 37/60) == "22:37"


def test_start_of_day():
    times = (
        ("Mon Jun 26 22:31:40 1950", "Mon Jun 26 00:00:00 1950"),
        ("Tue Jun 27 03:15:51 1950", "Tue Jun 27 00:00:00 1950"),
        ("Fri Jun 29 00:00:00 1950", "Fri Jun 29 00:00:00 1950"),
        ("Sat Jun 30 23:59:59 1950", "Sat Jun 30 00:00:00 1950"),
    )
    for t in times:
        assert time.asctime(start_of_day(time.strptime(t[0]))) == t[1]


def test_end_of_day():
    times = (
        ("Mon Jun 26 22:31:40 1950", "Mon Jun 26 23:59:60 1950"),
        ("Tue Jun 27 03:15:51 1950", "Tue Jun 27 23:59:60 1950"),
        ("Fri Jun 29 00:00:00 1950", "Fri Jun 29 23:59:60 1950"),
        ("Sat Jun 30 23:59:59 1950", "Sat Jun 30 23:59:60 1950"),
    )
    for t in times:
        assert time.asctime(end_of_day(time.strptime(t[0]))) == t[1]
