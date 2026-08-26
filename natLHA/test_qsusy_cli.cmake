cmake_minimum_required(VERSION 3.10)

foreach(required CLI SOURCE WORK)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "missing required test argument: ${required}")
    endif()
endforeach()

function(extract_field_token text field output)
    string(REGEX MATCH
        "(^|[\n ])${field} ([^\n ]+)([\n ]|$)"
        field_match "${text}")
    if(NOT field_match)
        message(FATAL_ERROR "missing ${field} field: ${text}")
    endif()
    set(${output} "${CMAKE_MATCH_2}" PARENT_SCOPE)
endfunction()

function(require_requested_spacing text field)
    extract_field_token("${text}" "${field}" declared_spacing)
    if(NOT "${declared_spacing}" STREQUAL "0.05"
            AND NOT "${declared_spacing}" STREQUAL "0.050000000000000003")
        message(FATAL_ERROR
            "${field} did not equal requested h=0.05: ${text}")
    endif()
endfunction()

function(require_bounded_observed_spacing text)
    extract_field_token("${text}" "max_observed_dlogQ" observed_spacing)
    if("${observed_spacing}" STREQUAL "0.050000000000000003")
        return()
    endif()
    if(NOT "${observed_spacing}"
            MATCHES "^(0|[1-9][0-9]*)(\\.[0-9]+)?$")
        message(FATAL_ERROR
            "observed spacing was not a canonical nonnegative decimal: ${text}")
    endif()
    if(NOT "${observed_spacing}"
            MATCHES "^0(\\.0([0-4][0-9]*|50*)?)?$")
        message(FATAL_ERROR "iteration exceeded non-default h: ${text}")
    endif()
endfunction()

file(MAKE_DIRECTORY "${WORK}")
set(batch "${WORK}/batch.txt")
file(WRITE "${batch}" "${SOURCE}\n")

execute_process(
    COMMAND "${CLI}" --slha "${SOURCE}" --qsusy-max-dlogq 0.05
    RESULT_VARIABLE single_result
    OUTPUT_VARIABLE single_stdout
    ERROR_VARIABLE single_stderr)
if(NOT single_result EQUAL 0)
    message(FATAL_ERROR
        "non-default single-point Q_SUSY run failed: ${single_result}; "
        "stdout=${single_stdout}; stderr=${single_stderr}")
endif()
string(REGEX MATCHALL "Q_SUSY iteration [^\n]*" iteration_lines "${single_stdout}")
list(LENGTH iteration_lines iteration_count)
if(iteration_count LESS 2)
    message(FATAL_ERROR
        "integration fixture did not exercise a joint re-search: ${single_stdout}")
endif()
foreach(line IN LISTS iteration_lines)
    require_requested_spacing("${line}" "max_dlogQ")
    require_bounded_observed_spacing("${line}")
endforeach()

execute_process(
    COMMAND "${CLI}" --batch "${batch}" --qsusy-max-dlogq 0.05
    RESULT_VARIABLE batch_result
    OUTPUT_VARIABLE batch_stdout
    ERROR_VARIABLE batch_stderr)
if(NOT batch_result EQUAL 0)
    message(FATAL_ERROR
        "non-default batch Q_SUSY run failed: ${batch_result}; "
        "stdout=${batch_stdout}; stderr=${batch_stderr}")
endif()
if(NOT batch_stdout MATCHES "^# ok Delta_EW Q_SUSY logQ_GUT mZ2 slha_path\n")
    message(FATAL_ERROR "batch stdout schema changed: ${batch_stdout}")
endif()
require_requested_spacing("${batch_stderr}" "q_susy_max_dlogq")

execute_process(
    COMMAND "${CLI}" --batch "${batch}" --qsusy-max-dlogq 0.05
            --qsusy-audit --digits 17
    RESULT_VARIABLE audit_result
    OUTPUT_VARIABLE audit_stdout
    ERROR_VARIABLE audit_stderr)
if(NOT audit_result EQUAL 0)
    message(FATAL_ERROR
        "structured Q_SUSY audit run failed: ${audit_result}; "
        "stdout=${audit_stdout}; stderr=${audit_stderr}")
endif()
string(REGEX MATCHALL "[^\n]+" audit_lines "${audit_stdout}")
list(LENGTH audit_lines audit_line_count)
if(NOT audit_line_count EQUAL 2)
    message(FATAL_ERROR "structured Q_SUSY audit emitted extra rows: ${audit_stdout}")
endif()
list(GET audit_lines 0 audit_header)
if(NOT audit_header STREQUAL
        "# ok Delta_EW Q_SUSY logQ_GUT mZ2 Q_SUSY_search_ok Q_SUSY_roots Q_SUSY_scan_complete Q_SUSY_searches Q_SUSY_search_logQ slha_path")
    message(FATAL_ERROR "structured Q_SUSY audit header changed: ${audit_stdout}")
endif()
list(GET audit_lines 1 audit_row)
string(REGEX REPLACE " +" ";" audit_fields "${audit_row}")
list(LENGTH audit_fields audit_field_count)
if(NOT audit_field_count EQUAL 11)
    message(FATAL_ERROR "structured Q_SUSY audit row is not rectangular: ${audit_row}")
endif()
list(GET audit_fields 5 audit_search_ok)
list(GET audit_fields 6 audit_roots)
list(GET audit_fields 7 audit_complete)
list(GET audit_fields 8 audit_searches)
list(GET audit_fields 9 audit_log_q)
list(GET audit_fields 10 audit_path)
if(NOT audit_search_ok STREQUAL "1"
        OR NOT audit_roots STREQUAL "1"
        OR NOT audit_complete STREQUAL "1"
        OR NOT audit_searches STREQUAL "${iteration_count}"
        OR NOT audit_log_q MATCHES "^[0-9]+\\.[0-9]+e[+-][0-9]+$"
        OR NOT audit_path STREQUAL "${SOURCE}")
    message(FATAL_ERROR "structured Q_SUSY audit fields are wrong: ${audit_row}")
endif()
require_requested_spacing("${audit_stderr}" "q_susy_max_dlogq")

set(missing_source "${WORK}/intentionally_absent_qsusy_audit_fixture.slha")
if(EXISTS "${missing_source}")
    message(FATAL_ERROR "missing-spectrum fixture unexpectedly exists: ${missing_source}")
endif()
set(failure_batch "${WORK}/failure-batch.txt")
file(WRITE "${failure_batch}" "${SOURCE}\n${missing_source}\n")
execute_process(
    COMMAND "${CLI}" --batch "${failure_batch}" --qsusy-max-dlogq 0.05
            --qsusy-audit --digits 17
    RESULT_VARIABLE failure_result
    OUTPUT_VARIABLE failure_stdout
    ERROR_VARIABLE failure_stderr)
if(NOT failure_result EQUAL 2)
    message(FATAL_ERROR
        "failed audit row did not produce status 2: ${failure_result}; "
        "stdout=${failure_stdout}; stderr=${failure_stderr}")
endif()
string(REGEX MATCHALL "[^\n]+" failure_lines "${failure_stdout}")
list(LENGTH failure_lines failure_line_count)
if(NOT failure_line_count EQUAL 3)
    message(FATAL_ERROR "failed audit run did not emit two rows: ${failure_stdout}")
endif()
list(GET failure_lines 2 failure_row)
string(REGEX REPLACE " +" ";" failure_fields "${failure_row}")
list(LENGTH failure_fields failure_field_count)
if(NOT failure_field_count EQUAL 11)
    message(FATAL_ERROR "failed audit row is not rectangular: ${failure_row}")
endif()
list(GET failure_fields 0 failure_ok)
list(GET failure_fields 5 failure_search_ok)
list(GET failure_fields 6 failure_roots)
list(GET failure_fields 7 failure_complete)
list(GET failure_fields 8 failure_searches)
list(GET failure_fields 9 failure_log_q)
list(GET failure_fields 10 failure_path)
if(NOT failure_ok STREQUAL "0"
        OR NOT failure_search_ok STREQUAL "0"
        OR NOT failure_roots STREQUAL "-1"
        OR NOT failure_complete STREQUAL "0"
        OR NOT failure_searches STREQUAL "0"
        OR NOT failure_log_q MATCHES "^0\\.0+e[+]00$"
        OR NOT failure_path STREQUAL "${missing_source}")
    message(FATAL_ERROR "failed audit row lost its unknown-count sentinel: ${failure_row}")
endif()
if(NOT failure_stderr MATCHES "point failed: ${missing_source}:")
    message(FATAL_ERROR "failed audit row lost its stderr diagnostic: ${failure_stderr}")
endif()
