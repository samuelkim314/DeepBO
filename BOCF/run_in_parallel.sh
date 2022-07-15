#!/bin/sh
# Start many worker threads, each of which will save output to a separate file
# in a directory specified within the python script.

for n in {1..5}
do
	# Add a job, which is to run peter_example_worker with argument n, to the long NBS queue.
	jsub "individual_worker.nbs $n"
done
