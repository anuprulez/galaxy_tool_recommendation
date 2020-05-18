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

query_workflow-connections() { ## [--all]: The connections of tools, from output to input, in the latest (or all) versions of user workflows (tool_predictions)
	handle_help "$@" <<-EOF
		This is used by the usegalaxy.eu tool prediction workflow, allowing for building models out of tool connections in workflows.
		    $ gxadmin query workflow-connections
		     wf_id |     wf_updated      | in_id |      in_tool      | in_tool_v | out_id |     out_tool      | out_tool_v | published | deleted | has_errors
		    -------+---------------------+-------+-------------------+-----------+--------+-------------------+----------------------------------------------
		         3 | 2013-02-07 16:48:00 |     5 | Grep1             | 1.0.1     |     12 |                   |            |    f      |    f    |    f
		         3 | 2013-02-07 16:48:00 |     6 | Cut1              | 1.0.1     |      7 | Remove beginning1 | 1.0.0      |    f      |    f    |    f
		         3 | 2013-02-07 16:48:00 |     7 | Remove beginning1 | 1.0.0     |      5 | Grep1             | 1.0.1      |    f      |    f    |    f
		         3 | 2013-02-07 16:48:00 |     8 | addValue          | 1.0.0     |      6 | Cut1              | 1.0.1      |    t      |    f    |    f
		         3 | 2013-02-07 16:48:00 |     9 | Cut1              | 1.0.1     |      7 | Remove beginning1 | 1.0.0      |    f      |    f    |    f
		         3 | 2013-02-07 16:48:00 |    10 | addValue          | 1.0.0     |     11 | Paste1            | 1.0.0      |    t      |    f    |    f
		         3 | 2013-02-07 16:48:00 |    11 | Paste1            | 1.0.0     |      9 | Cut1              | 1.0.1      |    f      |    f    |    f
		         3 | 2013-02-07 16:48:00 |    11 | Paste1            | 1.0.0     |      8 | addValue          | 1.0.0      |    t      |    t    |    f
		         4 | 2013-02-07 16:48:00 |    13 | cat1              | 1.0.0     |     18 | addValue          | 1.0.0      |    t      |    f    |    f
		         4 | 2013-02-07 16:48:00 |    13 | cat1              | 1.0.0     |     20 | Count1            | 1.0.0      |    t      |    t    |    f
	EOF

	read -r -d '' wf_filter <<-EOF
	WHERE
		workflow.id in (
			SELECT
			 workflow.id
			FROM
			 stored_workflow
			LEFT JOIN
			 workflow on stored_workflow.latest_workflow_id = workflow.id
		)
	EOF
	if [[ $1 == "--all" ]]; then
		wf_filter=""
	fi

	read -r -d '' QUERY <<-EOF
		SELECT
                        workflow.id as wf_id,
                        workflow.update_time as wf_updated,
                        ws_in.id as in_id,
                        ws_in.tool_id as in_tool,
                        ws_in.tool_version as in_tool_v,
                        ws_out.id as out_id,
                        ws_out.tool_id as out_tool,
                        ws_out.tool_version as out_tool_v,
                        sw.published as published,
                        sw.deleted as deleted,
                        workflow.has_errors as has_errors
		FROM workflow_step_connection wfc
		LEFT JOIN workflow_step ws_in ON ws_in.id = wfc.output_step_id
		LEFT JOIN workflow_step_input wsi ON wfc.input_step_input_id = wsi.id
		LEFT JOIN workflow_step ws_out ON ws_out.id = wsi.workflow_step_id
		LEFT JOIN workflow_output as wo ON wsi.workflow_step_id = wfc.output_step_id
		LEFT JOIN workflow on ws_in.workflow_id = workflow.id
		LEFT JOIN stored_workflow as sw on sw.latest_workflow_id = workflow.id
		$wf_filter
	EOF
}
