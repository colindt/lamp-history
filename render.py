#!/usr/bin/env python3
#coding: utf-8

import sys
import os
import time
import datetime
import math
import enum
import collections

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm
import numpy as np

from pprint import pprint as pp

sys.stdout.reconfigure(line_buffering=True) # auto-flush print()


colormap = mpl.cm.get_cmap("viridis").reversed()
background_color = (192, 192, 192)
grid_line_color = (192, 96, 96)

seconds_per_day = 24 * 60 * 60

Trend = enum.IntEnum("Trend", {"rising":1, "falling":-1, "steady":0})
OnOff = enum.Enum("OnOff", {"on":1, "off":0})


class Event:
    def __init__(self, time, pin, state, delay, line):
        self.time  = time
        self.pin   = pin
        self.state = state
        self.delay = delay
        self.line  = line
    
    def __repr__(self):
        return f"Event([{time.asctime(self.time)}] {self.state})"


class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end   = end
    
    def __repr__(self):
        return f"Segment([{time.asctime(self.start)}] -> [{time.asctime(self.end)}])"
    
    def as_tuple(self):
        return (self.start, self.end)


class TrendSegment (Segment):
    def __init__(self, start, end, trend, start_value=None, end_value=None):
        self.start       = start
        self.end         = end
        self.trend       = trend
        self.start_value = start_value
        self.end_value   = end_value
    
    def __repr__(self):
        return f"TrendSegment([{time.asctime(self.start)}] -> [{time.asctime(self.end)}] {self.trend.name} {self.start_value} -> {self.end_value})"
    
    def as_tuple(self):
        return (self.start, self.end, self.trend, self.start_value, self.end_value)



def main(outfolder, infiles):
    print(f"input: {infiles}")
    print(f"output: {outfolder}/")

    os.makedirs(outfolder, exist_ok=True)

    events = load_events_files(infiles)

    off_segments, on_segments = make_segments(events)
    split_off_segments = split_segments(off_segments)
    split_on_segments = split_segments(on_segments)

    draw_chart_frame(*draw_segment_chart(split_off_segments)).save(f"{outfolder}/history.png")

    lut = colormap_to_lut(colormap)

    trend_segments = {}
    for days,name in ((1, "daily"), (7, "weekly"), (14, "fortnightly"), (30, "monthly")):
        print()
        trend_segments[days] = calc_trend_segment_values(split_segments(make_trend_segments(events, days * seconds_per_day)))
        draw_chart_frame(*draw_trend_chart(trend_segments[days], days * seconds_per_day, lut)).save(f"{outfolder}/{name}_trend.png")

    print()
    cohesion_segments = calc_trend_segment_values(split_segments(make_cohesion_segments(trend_segments[1])), seconds_per_day)
    with open(f"{outfolder}/cohesion_segments.txt", "w") as f:
        for s in cohesion_segments:
            f.write(str(s))
            f.write("\n")
    #draw_chart_frame(*draw_cohesion_chart(trend_segments[1], lut)).save(f"{outfolder}/cohesion.png")
    draw_chart_frame(*draw_trend_chart(cohesion_segments, seconds_per_day, lut, normalized=False)).save(f"{outfolder}/cohesion.png")

    print()
    figs = draw_plots(events, off_segments, on_segments, trend_segments)
    for i,fig in enumerate(figs):
        fig.savefig(f"{outfolder}/plots{i+1}.png")
        fig.savefig(f"{outfolder}/plots{i+1}.pdf")


def load_events_files(infiles):
    print("loading events...")

    events = []

    for fname in infiles:
        with open(fname) as f:
            for line in f:
                if line.strip() == "":
                    continue
                line = line.replace("\t", " ")
                event_time, rest = line[1:].split("]")
                pin, rest = rest.strip().split(" ", 1)
                state, rest = rest.strip().split(" ", 1)

                events.append(Event(
                    time  = time.strptime(event_time),
                    pin   = int(pin),
                    state = OnOff.on if state == "on" else OnOff.off,
                    delay = float(rest.strip().split()[0]),
                    line  = line.strip()
                ))

    return events


def make_segments(events):
    print("making segments...")

    off_segments = []
    on_segments = []
    state = events[0].state
    segment_start = events[0]
    for e in events:
        t1 = segment_start.time if segment_start else None
        t2 = e.time

        if state == OnOff.on:
            if e.state == OnOff.on:
                if segment_start:  #pragma: no cover
                    print("warning: state transition from on to on:", segment_start.time, e.time)
            elif e.state == OnOff.off:
                if t1:
                    on_segments.append(Segment(t1, t2))

                segment_start = e
                state = OnOff.off
            else:  #pragma: no cover
                raise ValueError("invalid event state '{}'".format(e["state"]))
        elif state == OnOff.off:
            t1 = segment_start.time
            t2 = e.time

            if e.state == OnOff.on:
                off_segments.append(Segment(t1, t2))

                segment_start = e
                state = OnOff.on
            elif e.state == OnOff.off:  #pragma: no cover
                print("warning: state transition from off to off", t1, t2)
            else:  #pragma: no cover
                raise ValueError("invalid event state '{}'".format(e.state))
        else:  #pragma: no cover
            raise ValueError("invalid state machine state '{}'".format(state))

    return (off_segments, on_segments)


def split_segments(segments):
    # split segments that span day boundaries into two segments that don't
    print("splitting segments...")

    new_segments = []

    for s in segments:
        date1 = time_to_date(s.start)
        date2 = time_to_date(s.end)
        if date1 == date2:
            new_segments.append(s)
        else:
            segment1 = type(s)(*(s.start, end_of_day(s.start)) + s.as_tuple()[2:])
            segment2 = type(s)(*(start_of_day(s.end), s.end) + s.as_tuple()[2:])

            new_segments.append(segment1)
            new_segments.append(segment2)

    return new_segments


def make_trend_segments(events, interval=seconds_per_day):
    print("making trend segments...")

    def state_transition_function(head_state, tail_state):
        if head_state == OnOff.on and tail_state == OnOff.off:
            return Trend.falling
        elif head_state == OnOff.off and tail_state == OnOff.on:
            return Trend.rising
        else:
            return Trend.steady

    return window_walk(events, "time", "state", OnOff.on, state_transition_function, interval)


def make_cohesion_segments(trend_segments, interval=seconds_per_day):
    print("making cohesion segments...")
    
    def state_transition_function(head_state, tail_state):
        if head_state == Trend.steady and tail_state != Trend.steady:
            return Trend.rising
        elif head_state != Trend.steady and tail_state == Trend.steady:
            return Trend.falling
        else:
            return Trend.steady
    
    # tack on a fake segment so that the last segment actually gets counted
    # note that we call window_walk with state_attr="start", so the end time of the last segment never gets counted
    walk_segments = trend_segments + [TrendSegment(trend_segments[-1].end, None, None)]
    
    return window_walk(walk_segments, "start", "trend", Trend.steady, state_transition_function, interval)


def window_walk(events, time_attr, state_attr, intial_tail_state, state_transition_function, interval=seconds_per_day):
    # two markers, `head` and `tail` walk along the timeline `interval` seconds apart. 

    window_segments = []
    
    head = time.mktime(getattr(events[0], time_attr))
    tail = head - interval
    
    head_next_index = 1
    tail_next_index = 0
    
    head_state = getattr(events[0], state_attr)
    tail_state = intial_tail_state
    
    state = state_transition_function(head_state, tail_state)
    
    prev_head = head

    while head_next_index < len(events):
        prev_head = head

        # calculate time between current markers and their next state transition
        head_diff = time.mktime(getattr(events[head_next_index], time_attr)) - head
        tail_diff = time.mktime(getattr(events[tail_next_index], time_attr)) - tail

        if head_diff > tail_diff:
            # if `tail` is closer to a state transition, move up to it
            head += tail_diff
            tail += tail_diff

            # record the state transition and look ahead to the next `tail` transition
            tail_state = getattr(events[tail_next_index], state_attr)
            tail_next_index += 1
        elif head_diff < tail_diff:
            # if `head` is closer to a state transition, move up to it
            head += head_diff
            tail += head_diff

            # record the state transition and look ahead to the next `head` transition
            head_state = getattr(events[head_next_index], state_attr)
            head_next_index += 1
        else:
            # if `head` and `tail` reach a state transition at the same time, move up to them
            head += head_diff
            tail += head_diff

            # record the state transitions and look ahead to the next transitions
            head_state = getattr(events[head_next_index], state_attr)
            tail_state = getattr(events[tail_next_index], state_attr)
            head_next_index += 1
            tail_next_index += 1

        # timestamps of segment to add
        start = time.localtime(prev_head)
        end = time.localtime(head)

        window_segments.append(TrendSegment(start, end, state))

        state = state_transition_function(head_state, tail_state)

    return window_segments


def calc_trend_segment_values(trend_segments, inital_value=0):
    print("caluclating trend segment values...")

    new_trend_segments = []

    prev_end_value = inital_value

    for s in trend_segments:
        segment_length = time.mktime(s.end) - time.mktime(s.start) # dst changes introduce an off-by-one error here and I'm still not sure why
        if s.start.tm_isdst == 0 and s.end.tm_isdst == 1: # fix for ^that
            print("DST fix: {} ({}); {} ({})".format(time.asctime(s.start), s.start.tm_isdst, time.asctime(s.end), s.end.tm_isdst))
            segment_length += 1

        start_value = prev_end_value
        end_value = start_value + (segment_length * s.trend)
        new_trend_segments.append(TrendSegment(s.start, s.end, s.trend, start_value, end_value))

        prev_end_value = end_value

    return new_trend_segments


# draw_function(draw, segment, coords, setup, lut, **kwargs) -> no return value
# setup_function(segments, **kwargs) -> (setup_for_draw_function, key_min, key_max)
# `kwargs` of `draw_chart()` are passed along to both functions
def draw_chart(segments, draw_function, setup_function=(lambda s,**k:(None,None,None)), lut=None, width=720, day_height=6, **kwargs):
    first_date = time_to_date(segments[0].start)
    last_date  = time_to_date(segments[-1].end)

    day_count = (last_date - first_date).days + 1

    im = Image.new("RGB", (width, day_height * day_count), (255,255,255))
    draw = ImageDraw.Draw(im)

    setup, key_min, key_max = setup_function(segments, **kwargs)

    for s in segments:
        y1 = (time_to_date(s.start) - first_date).days * day_height

        seconds_from_midnight = ((s.start.tm_hour * 60) + s.start.tm_min) * 60 + s.start.tm_sec
        x1 = seconds_from_midnight * width / seconds_per_day

        # this messes up for daylight savings changes because the w calculation just
        # calculates the difference in seconds without regard for the missing/extra 2am hour
        w = max(1, (time.mktime(s.end) - time.mktime(s.start)) * width / seconds_per_day)

        x2 = x1 + w
        y2 = y1 + day_height

        draw_function(draw, s, (round(x1), y1, round(x2)-1, y2-1), setup, lut, **kwargs)

    return (im, day_height, day_count, segments[0].start, lut, key_min, key_max)


def draw_segment_chart(off_segments, width=720, day_height=6):
    print("drawing segment chart...")

    def draw_function(draw, segment, coords, setup, lut):
        draw.rectangle(coords, fill=(0,0,0), outline=None, width=0)

    return draw_chart(off_segments, draw_function, width=width, day_height=day_height)



def draw_trend_chart(trend_segments, trend_interval, lut, width=720, day_height=6, normalized=True):
    print("drawing trend chart...")

    def setup_function(segments, trend_interval, normalized):
        if normalized:
            trend_max = 0
            trend_min = trend_interval
            for s in segments:
                trend_max = max(trend_max, s.start_value, s.end_value)

                if time.mktime(s.start) >= time.mktime(segments[0].start) + trend_interval: # don't update minumum on incomplete averages
                    trend_min = min(trend_min, s.start_value, s.end_value)

        else:
            trend_max = trend_interval
            trend_min = 0
        
        key_min = trend_min / trend_interval
        key_max = trend_max / trend_interval

        setup = {
            "start" : segments[0].start,
            "trend_min" : trend_min,
            "trend_max" : trend_max,
        }
        return (setup, key_min, key_max)


    def draw_function(draw, segment, coords, setup, lut, **kwargs):
        if time.mktime(segment.start) < time.mktime(setup["start"]) + kwargs["trend_interval"]:
            return
        
        trend_min = setup["trend_min"]
        trend_max = setup["trend_max"]

        start_value = (segment.start_value - trend_min) / (trend_max - trend_min)
        end_value   = (segment.end_value   - trend_min) / (trend_max - trend_min)

        draw_gradient(draw, *coords, start_value, end_value, lut)


    return draw_chart(trend_segments, draw_function, setup_function, lut, width=width, day_height=day_height, trend_interval=trend_interval, normalized=normalized)



def draw_cohesion_chart(trend_segments, lut, width=720, day_height=6):
    print("drawing cohesion chart...")

    def setup_function(segments):
        return (None, 0, 1)


    def draw_function(draw, segment, coords, setup, lut):
        if segment.trend == Trend.rising:
            fill = (128,255,128)
        elif segment.trend == Trend.falling:
            fill = (255,128,128)
        else:
            fill = (0,0,0)
        draw.rectangle(coords, fill=fill, outline=None, width=0)


    return draw_chart(trend_segments, draw_function, setup_function, width=width, day_height=day_height)



def colormap_to_lut(colormap):
    lut = [list(colormap(i)[:-1]) for i in np.linspace(0, 1, 256)]

    if type(colormap) == mpl.colors.ListedColormap:
        assert len(colormap.colors) == 256
        assert lut == colormap.colors

    lut = [tuple(round(255*i) for i in c) for c in lut]
    return lut


def draw_gradient(draw, x1, y1, x2, y2, start_value, end_value, lut):
    if start_value == end_value:
        draw.rectangle((x1, y1, x2, y2), fill=lut[round(255 * start_value)], outline=None, width=0)
    else:
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        for x in range(x1, x2+1):
            progress = (x - x1) / w
            value = progress * (end_value - start_value) + start_value
            if value > 1:
                color = (255, 255, 255)
                print(value, color)
            elif value < 0:
                color = (0, 0, 0)
                print(value, color)
            else:
                color = lut[round(255 * value)]

            draw.rectangle((x, y1, x, y2), fill=color, outline=None, width=0)


def draw_chart_frame(chart_image, day_height, day_count, start_time, lut=None, key_min=None, key_max=None):
    print("drawing chart frame...")

    first_date = time_to_date(start_time)

    font = ImageFont.truetype("arial")

    chart_draw = ImageDraw.Draw(chart_image)

    grid_color_lookup = {} # matplotlib colorspace conversion is slow, so we're only doing each conversion once

    # hour grid
    for i in range(24):
        x = i * 60 * 60 * chart_image.width / seconds_per_day
        for y in range(0, day_height * day_count, 2):
            c = chart_image.getpixel((x, y))
            if c not in grid_color_lookup:
                grid_color_lookup[c] = grid_color(c)
            color = grid_color_lookup[c]
            chart_draw.point((x, y), color)

    # week grid
    max_date_textlength = 0
    for i in range(6 - start_time.tm_wday, day_count, 7):
        d = first_date + datetime.timedelta(i)
        max_date_textlength = max(max_date_textlength, chart_draw.textlength(str(d), font))

        y = i * day_height
        for x in range(0, chart_image.width, 2):
            # don't invert twice (handle grid line intersections)
            if x % (60 * 60 * chart_image.width / seconds_per_day) == 0:
                continue
            c = chart_image.getpixel((x, y))
            if c not in grid_color_lookup:
                grid_color_lookup[c] = grid_color(c)
            color = grid_color_lookup[c]
            chart_draw.point((x, y), color)

    text_vbox = 18
    text_outer_hpad = 8
    text_inner_hpad = 4
    text_inner_vpad = 4

    top_padding = text_vbox
    left_padding = math.ceil(max_date_textlength) + text_outer_hpad
    bottom_padding = text_vbox
    right_padding = text_outer_hpad

    if lut:
        key_height = 32
        key_interpad = 8
        bottom_padding += key_interpad + key_height + text_vbox

    width = chart_image.width + left_padding + right_padding
    height = chart_image.height + top_padding + bottom_padding
    im = Image.new("RGB", (width, height), background_color)
    im.paste(chart_image, (left_padding, top_padding))
    draw = ImageDraw.Draw(im)

    # hour labels
    for i in range(0, 24, 2):
        x = (i * 60 * 60 * chart_image.width / seconds_per_day) + left_padding
        if i == 0:
            text = "midnight"
        elif 1 <= i <= 11:
            text = "{}am".format(i)
        elif i == 12:
            text = "noon"
        else:
            text = "{}pm".format(i-12)
        draw.text((x, top_padding - text_inner_vpad), text, (0,0,0), font, "ms")
        draw.text((x, height - bottom_padding + text_inner_vpad), text, (0,0,0), font, "mt")

    # week labels
    for i in range(6 - start_time.tm_wday, day_count, 7):
        d = first_date + datetime.timedelta(i)
        draw.text((left_padding - text_inner_hpad, i * day_height + top_padding), str(d), (0,0,0), font, "rt")

    # color key
    if lut:
        x1 = left_padding
        y1 = im.height - text_vbox - key_height
        x2 = im.width - right_padding - 1
        y2 = im.height - text_vbox - 1
        draw_gradient(draw, x1, y1, x2, y2, 0, 1, lut)

        draw.text((x1, y2 + text_inner_vpad), format_hours(key_min*24), (0,0,0), font, "lt")
        draw.text((x2, y2 + text_inner_vpad), format_hours(key_max*24), (0,0,0), font, "rt")

        # grid and labels
        hours_delta = (key_max - key_min) * 24
        pixels_per_hour = chart_image.width / hours_delta
        first_hour = math.ceil(key_min * 24)
        if abs(first_hour - key_min * 24) < 1e-8: # fix for off-by-one error when starting on an integer
            print("integer start fix")
            first_hour += 1
        last_hour = math.floor(key_max * 24)
        start_offset_hours = 1 - (key_min * 24) % 1
        start_offset_pixels = start_offset_hours * pixels_per_hour + left_padding

        h_ = list(range(last_hour - first_hour + 1))
        marker_x = [int(round(i * pixels_per_hour + start_offset_pixels)) for i in h_]

        for i,x in enumerate(marker_x):
            for y in range(y1, y2, 2):
                c = im.getpixel((x, y))
                if c not in grid_color_lookup:
                    grid_color_lookup[c] = grid_color(c)
                color = grid_color_lookup[c]
                draw.point((x, y), color)
            draw.text((x, y2 + text_inner_vpad), str(i + first_hour), (0,0,0), font, "mt")

    return im


def grid_color(c):
    # figure out what color a grid line should be based
    # on what it's going on top of for good contrast
    h, s, v = mpl.colors.rgb_to_hsv([i/255 for i in c])

    h += 0.5
    if h > 1:
        h -= 1

    v += 0.5
    if v > 1:
        v -= 1

    return tuple(int(round(i*255)) for i in mpl.colors.hsv_to_rgb((h, s, v)))


def format_hours(t):
    if t < 0:  #pragma: no cover
        print("WARNING: negative t: {}. Setting to 0".format(t))
        t = 0
    hours = int(t)
    minutes = t % 1 * 60
    return "{:2.0f}:{:02.0f}".format(hours, minutes)


def draw_plots(events, off_segments, on_segments, trend_segments):
    print("drawing plots...")

    advanced_plots = True

    # page 1
    w = 8.5
    h = 11

    if advanced_plots:
        fig1, axs = plt.subplots(3, 2, figsize=(w,h), tight_layout=True)
    else:
        fig1, axs = plt.subplots(2, 1, figsize=(w,h), tight_layout={"rect":(1.25/w, 0.6/h, 7.25/w, 10.5/h), "h_pad":3}, squeeze=False)

    off_segment_lengths = segment_length_hours(off_segments)
    on_segment_lengths = segment_length_hours(on_segments)

    bins = [i/2 for i in range(0, math.ceil(max(off_segment_lengths) * 2) + 1)] # bins in 30 minute intervals
    axs[0,0].hist(off_segment_lengths, weights=off_segment_lengths, bins=bins, density=True)
    axs[0,0].set_title("Weighted Off Segments")
    axs[0,0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axs[0,0].set_xlabel("hours")

    if advanced_plots:
        ax = axs[0,1]
    else:
        ax = axs[1,0]
    bins = range(0, math.ceil(max(on_segment_lengths)) + 1, 2) # bins in 2 hour intervals
    ax.hist(on_segment_lengths, weights=on_segment_lengths, bins=bins, density=True)
    ax.set_title("Weighted On Segments")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    ax.set_xlabel("hours")

    if advanced_plots:
        n = min(len(off_segments), len(on_segments))

        if off_segments[0].end == on_segments[0].start: # starts with off segment
            print("off segment first")
            x1 = off_segment_lengths[:n]
            y1 = on_segment_lengths[:n]
            c1 = [time.mktime(i.start) for i in on_segments[:n]]

            x2 = on_segment_lengths[:n-1]
            y2 = off_segment_lengths[1:n]
            c2 = [time.mktime(i.start) for i in off_segments[1:n]]
        elif on_segments[0].end == off_segments[0].start: # starts with on segment
            print("WARNING: on segment first (UNTESTED)")
            x1 = off_segment_lengths[:n-1]
            y1 = on_segment_lengths[1:n]
            c1 = [time.mktime(i.start) for i in on_segments[1:n]]

            x2 = on_segment_lengths[:n]
            y2 = off_segment_lengths[:n]
            c2 = [time.mktime(i.start) for i in off_segments[:n]]
        else:
            print("WARNING: first segments don't line up")

        scattersize = 8

        axs[1,0].scatter(x1, y1, s=scattersize, c=c1)
        axs[1,0].set_title("Off vs Next On Segment")
        axs[1,0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        axs[1,0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
        axs[1,0].set_xlabel("Off Segment Length")
        axs[1,0].set_ylabel("Next On Segment Length")
        axs[1,0].set_xlim(axs[0,0].get_xlim())

        axs[1,1].scatter(x2, y2, s=scattersize, c=c2)
        axs[1,1].set_title("On vs Next Off Segment")
        axs[1,1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
        axs[1,1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        axs[1,1].set_xlabel("On Segment Length")
        axs[1,1].set_ylabel("Next Off Segment Length")
        axs[1,1].set_xlim(axs[0,1].get_xlim())

        trend_values = {}
        for s in trend_segments[1]:
            trend_values[time.mktime(s.start)] = s.start_value

        x3 = [trend_values[time.mktime(i.start)]/(60*60) for i in off_segments]
        c3 = [time.mktime(i.start) for i in off_segments]
        axs[2,0].scatter(x3, off_segment_lengths, s=scattersize, c=c3)
        axs[2,0].set_title("Previous 24 Hours vs Off Segment Length")
        axs[2,0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
        axs[2,0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        axs[2,0].set_xlabel("Time Off in the Last Day")
        axs[2,0].set_ylabel("Subsequent Off Segment Length")

        x4 = [trend_values[time.mktime(i.start)]/(60*60) for i in on_segments]
        c4 = [time.mktime(i.start) for i in on_segments]
        axs[2,1].scatter(x4, on_segment_lengths, s=scattersize, c=c4)
        axs[2,1].set_title("Previous 24 Hours vs On Segment Length")
        axs[2,1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
        axs[2,1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
        axs[2,1].set_xlabel("Time Off in the Last Day")
        axs[2,1].set_ylabel("Subsequent On Segment Length")
    
    # page 2
    w = 11
    h = 8.5

    fig2, ax = plt.subplots(1, 1, figsize=(w,h), tight_layout=True)

    
    for k in trend_segments:
        x5 = [datetime.datetime.fromtimestamp(time.mktime(i.start)) for i in trend_segments[k]]
        y5 = [i.start_value/(60*60*k) for i in trend_segments[k]]
        ax.plot(x5, y5, linewidth=1)

    return (fig1, fig2)


def segment_length_hours(segments):
    segment_lengths = []
    for s in segments:
        t1 = struct_time_to_datetime(s.start)
        t2 = struct_time_to_datetime(s.end)
        d = t2 - t1
        hours = (d.days * 24) + (d.seconds / (60*60))
        segment_lengths.append(hours)

    return segment_lengths


def time_to_date(t):
    return datetime.date.fromtimestamp(time.mktime(t))


def start_of_day(t):
    return time.struct_time((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, t.tm_wday, t.tm_yday, -1))


def end_of_day(t):
    # 23:59:60 is a fake time (usually), but 23:59:59 introduces off-by-one errors beacuse we miss the last second of the day
    # (especially relevant to calc_trend_segment_values)
    return time.struct_time((t.tm_year, t.tm_mon, t.tm_mday, 23, 59, 60, t.tm_wday, t.tm_yday, -1))


def struct_time_to_datetime(t):
    return datetime.datetime.fromtimestamp(time.mktime(t))


if __name__ == "__main__":  #pragma: no cover
    if len(sys.argv) == 1:
        # with no arguments, read input.txt for a list of arguments
        with open("input.txt") as f:
            args = f.read().strip().split('\n')
        main(args[0], args[1:])
    else:
        # first argument output folder, remaining arguments input files in order
        main(sys.argv[1], sys.argv[2:])
