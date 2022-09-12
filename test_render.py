#!/usr/bin/env python3

import pytest
import time

from render import *


@pytest.fixture
def events():
    return load_events_files(["test_data/test.txt"])


@pytest.fixture
def segments(events):
    return make_segments(events)


def test_load_events_files(events):
    assert len(events) == 11
    assert events[0].state == OnOff.on
    assert events[0].time.tm_hour == 0

    assert events[1].state == OnOff.off
    assert events[1].time.tm_hour == 1

    assert events[2].state == OnOff.on
    assert events[2].time.tm_hour == 3
    
    assert events[10].state == OnOff.on
    assert events[10].time.tm_hour == 10


def test_make_segments(segments):
    off_segments, on_segments = segments
    
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
    

def test_split_segments(segments):
    off_segments, on_segments = segments

    split_off_segments = split_segments(off_segments)
    assert len(split_off_segments) == 6
    assert split_off_segments[0].end.tm_hour == 3
    assert split_off_segments[3].end.tm_hour == 23
    assert split_off_segments[3].end.tm_min == 59
    assert split_off_segments[3].end.tm_sec == 59
    assert split_off_segments[4].start.tm_hour == 0
    assert split_off_segments[4].start.tm_min == 0
    assert split_off_segments[4].start.tm_sec == 0
    assert split_off_segments[4].end.tm_hour == 3

    split_on_segments  = split_segments(on_segments)
    assert len(split_on_segments) == 6
    assert split_on_segments[0].end.tm_hour == 1
    assert split_on_segments[2].end.tm_hour == 23
    assert split_on_segments[2].end.tm_min == 59
    assert split_on_segments[2].end.tm_sec == 59
    assert split_on_segments[3].start.tm_hour == 0
    assert split_on_segments[3].start.tm_min == 0
    assert split_on_segments[3].start.tm_sec == 0


@pytest.mark.skip
def test_make_trend_segments():
    pass


@pytest.mark.skip
def test_make_cohesion_segments():
    pass


@pytest.mark.skip
def test_calc_trend_segment_values():
    pass


@pytest.mark.skip
def test_draw_segment_chart():
    pass


@pytest.mark.skip
def test_draw_trend_chart():
    pass


@pytest.mark.skip
def test_draw_cohesion_chart():
    pass


@pytest.mark.skip
def test_draw_plots():
    pass


def test_segment_length_hours(segments):
    off_segments, on_segments = segments
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
        ("Mon Jun 26 22:31:40 1950", "Mon Jun 26 23:59:59 1950"),
        ("Tue Jun 27 03:15:51 1950", "Tue Jun 27 23:59:59 1950"),
        ("Fri Jun 29 00:00:00 1950", "Fri Jun 29 23:59:59 1950"),
        ("Sat Jun 30 23:59:59 1950", "Sat Jun 30 23:59:59 1950"),
    )
    for t in times:
        assert time.asctime(end_of_day(time.strptime(t[0]))) == t[1]
