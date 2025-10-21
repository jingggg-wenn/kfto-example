#!/bin/bash

oc apply -f project.yaml
oc apply -f scc.yaml
sleep 5
oc apply -f .
