#!/bin/bash
# Create dir if not existing
mkdir -p data/

query_tool-popularity() { ## [months|24]: Most run tools by month
	handle_help "$@" <<-EOF
		See most popular tools by month
		    $ ./gxadmin query tool-popularity 1
		              tool_id          |   month    | count
		    ---------------------------+------------+-------
		     circos                    | 2019-02-01 |    20
		     upload1                   | 2019-02-01 |    12
		     require_format            | 2019-02-01 |     9
		     circos_gc_skew            | 2019-02-01 |     7
		     circos_wiggle_to_scatter  | 2019-02-01 |     3
		     test_history_sanitization | 2019-02-01 |     2
		     circos_interval_to_tile   | 2019-02-01 |     1
		     __SET_METADATA__          | 2019-02-01 |     1
		    (8 rows)
	EOF

	fields="count=2"
	tags="tool_id=0;month=1"

	months=${1:-24}

	read -r -d '' QUERY <<-EOF
		SELECT
			tool_id,
			date_trunc('month', create_time AT TIME ZONE 'UTC')::date as month,
			count(*)
		FROM job
		WHERE create_time > (now() AT TIME ZONE 'UTC' - '$months months'::interval)
		GROUP BY tool_id, month
		ORDER BY month desc, count desc
	EOF
}

query_workflow-connections() { ## [--all]: The connections of tools, from output to input, in the latest (or all) versions of user workflows
	handle_help "$@" <<-EOF
		This is used by the usegalaxy.eu tool prediction workflow, allowing for building models out of tool connections in workflows.
		    $ gxadmin query workflow-connections
		     wf_id |     wf_updated      | in_id |      in_tool      | in_tool_v | out_id |     out_tool      | out_tool_v
		    -------+---------------------+-------+-------------------+-----------+--------+-------------------+------------
		         3 | 2013-02-07 16:48:00 |     5 | Grep1             | 1.0.1     |     12 |                   |
		         3 | 2013-02-07 16:48:00 |     6 | Cut1              | 1.0.1     |      7 | Remove beginning1 | 1.0.0
		         3 | 2013-02-07 16:48:00 |     7 | Remove beginning1 | 1.0.0     |      5 | Grep1             | 1.0.1
		         3 | 2013-02-07 16:48:00 |     8 | addValue          | 1.0.0     |      6 | Cut1              | 1.0.1
		         3 | 2013-02-07 16:48:00 |     9 | Cut1              | 1.0.1     |      7 | Remove beginning1 | 1.0.0
		         3 | 2013-02-07 16:48:00 |    10 | addValue          | 1.0.0     |     11 | Paste1            | 1.0.0
		         3 | 2013-02-07 16:48:00 |    11 | Paste1            | 1.0.0     |      9 | Cut1              | 1.0.1
		         3 | 2013-02-07 16:48:00 |    11 | Paste1            | 1.0.0     |      8 | addValue          | 1.0.0
		         4 | 2013-02-07 16:48:00 |    13 | cat1              | 1.0.0     |     18 | addValue          | 1.0.0
		         4 | 2013-02-07 16:48:00 |    13 | cat1              | 1.0.0     |     20 | Count1            | 1.0.0
	EOF
