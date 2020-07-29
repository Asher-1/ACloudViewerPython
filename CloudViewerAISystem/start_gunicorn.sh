#!/usr/bin/env bash
gunicorn -c gun_conf.py app:app