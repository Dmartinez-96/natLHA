cmake_minimum_required(VERSION 3.10)

foreach(required CLI SOURCE WORK CUDA_ENABLED)
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
if(batch_stderr MATCHES "# backend requested" OR batch_stderr MATCHES "# cuda_profile")
    message(FATAL_ERROR
        "default CPU batch gained backend/profile stderr output: ${batch_stderr}")
endif()

execute_process(
    COMMAND "${CLI}" --batch "${batch}" --qsusy-max-dlogq 0.05
            --backend auto --backend-audit
    RESULT_VARIABLE backend_audit_result
    OUTPUT_VARIABLE backend_audit_stdout
    ERROR_VARIABLE backend_audit_stderr)
if(NOT backend_audit_result EQUAL 0)
    message(FATAL_ERROR
        "backend-audited batch failed: ${backend_audit_result}; "
        "stdout=${backend_audit_stdout}; stderr=${backend_audit_stderr}")
endif()
if(NOT backend_audit_stderr MATCHES "# backend requested")
    message(FATAL_ERROR
        "backend-audited batch lost its backend summary: ${backend_audit_stderr}")
endif()
if(CUDA_ENABLED AND NOT backend_audit_stderr MATCHES "# cuda_profile")
    message(FATAL_ERROR
        "CUDA-audited batch lost its stage profiles: ${backend_audit_stderr}")
endif()
string(REGEX MATCHALL "[^\n]+" backend_audit_lines "${backend_audit_stdout}")
list(LENGTH backend_audit_lines backend_audit_line_count)
if(NOT backend_audit_line_count EQUAL 2)
    message(FATAL_ERROR
        "backend audit emitted an unexpected row count: ${backend_audit_stdout}")
endif()
list(GET backend_audit_lines 0 backend_audit_header)
if(NOT backend_audit_header STREQUAL
        "# ok Delta_EW Q_SUSY logQ_GUT mZ2 backend_executed selected_backend candidate_tier final_tier adjudication_reasons cpu_adjudicated backend_audit_match slha_path")
    message(FATAL_ERROR "backend audit header changed: ${backend_audit_stdout}")
endif()
list(GET backend_audit_lines 1 backend_audit_row)
string(REGEX REPLACE " +" ";" backend_audit_fields "${backend_audit_row}")
list(LENGTH backend_audit_fields backend_audit_field_count)
if(NOT backend_audit_field_count EQUAL 13)
    message(FATAL_ERROR "backend audit row is not rectangular: ${backend_audit_row}")
endif()
list(GET backend_audit_fields 5 backend_executed)
list(GET backend_audit_fields 6 selected_backend)
list(GET backend_audit_fields 7 candidate_tier)
list(GET backend_audit_fields 8 final_tier)
list(GET backend_audit_fields 11 backend_audit_match)
list(GET backend_audit_fields 12 backend_audit_path)
if(NOT backend_executed STREQUAL "1"
        OR NOT selected_backend MATCHES "^(cpu|cuda)$"
        OR candidate_tier STREQUAL "none"
        OR final_tier STREQUAL "none"
        OR NOT backend_audit_path STREQUAL "${SOURCE}")
    message(FATAL_ERROR
        "executed backend audit lost its provenance: ${backend_audit_row}")
endif()
if(CUDA_ENABLED)
    if(NOT selected_backend STREQUAL "cuda"
            OR NOT backend_audit_match STREQUAL "1")
        message(FATAL_ERROR
            "CUDA build did not execute and match a CUDA audit: ${backend_audit_row}")
    endif()
else()
    if(NOT selected_backend STREQUAL "cpu"
            OR NOT backend_audit_match STREQUAL "-1")
        message(FATAL_ERROR
            "CPU build did not record automatic CPU fallback: ${backend_audit_row}")
    endif()
endif()

set(provenance_path "${WORK}/backend-provenance.tsv")
file(REMOVE "${provenance_path}")
execute_process(
    COMMAND "${CLI}" --batch "${batch}" --qsusy-max-dlogq 0.05
            --backend auto --backend-provenance-out "${provenance_path}"
    RESULT_VARIABLE provenance_result
    OUTPUT_VARIABLE provenance_stdout
    ERROR_VARIABLE provenance_stderr)
if(NOT provenance_result EQUAL 0)
    message(FATAL_ERROR
        "backend-provenance batch failed: ${provenance_result}; "
        "stdout=${provenance_stdout}; stderr=${provenance_stderr}")
endif()
if(NOT provenance_stdout MATCHES
        "^# ok Delta_EW Q_SUSY logQ_GUT mZ2 slha_path\n")
    message(FATAL_ERROR
        "backend provenance changed the primary row schema: ${provenance_stdout}")
endif()
if(provenance_stdout MATCHES "backend_executed")
    message(FATAL_ERROR
        "backend provenance leaked into the primary rows: ${provenance_stdout}")
endif()
file(READ "${provenance_path}" provenance_text)
string(REGEX MATCHALL "[^\n]+" provenance_lines "${provenance_text}")
list(LENGTH provenance_lines provenance_line_count)
if(NOT provenance_line_count EQUAL 2)
    message(FATAL_ERROR
        "backend provenance emitted an unexpected row count: ${provenance_text}")
endif()
list(GET provenance_lines 0 provenance_header)
if(NOT provenance_header STREQUAL
        "# backend_executed selected_backend candidate_tier final_tier adjudication_reasons cpu_adjudicated slha_path")
    message(FATAL_ERROR "backend provenance header changed: ${provenance_text}")
endif()
list(GET provenance_lines 1 provenance_row)
string(REGEX REPLACE " +" ";" provenance_fields "${provenance_row}")
list(LENGTH provenance_fields provenance_field_count)
if(NOT provenance_field_count EQUAL 7)
    message(FATAL_ERROR "backend provenance row is not rectangular: ${provenance_row}")
endif()
list(GET provenance_fields 0 provenance_executed)
list(GET provenance_fields 1 provenance_backend)
list(GET provenance_fields 2 provenance_candidate)
list(GET provenance_fields 3 provenance_final)
list(GET provenance_fields 6 provenance_source)
if(NOT provenance_executed STREQUAL "1"
        OR provenance_candidate STREQUAL "none"
        OR provenance_final STREQUAL "none"
        OR NOT provenance_source STREQUAL "${SOURCE}")
    message(FATAL_ERROR
        "backend provenance lost executed-row diagnostics: ${provenance_row}")
endif()
if(CUDA_ENABLED)
    if(NOT provenance_backend STREQUAL "cuda")
        message(FATAL_ERROR
            "CUDA provenance run did not select CUDA: ${provenance_row}")
    endif()
else()
    if(NOT provenance_backend STREQUAL "cpu")
        message(FATAL_ERROR
            "CPU-only automatic provenance run did not record fallback: ${provenance_row}")
    endif()
endif()

set(unavailable_provenance_path "${WORK}/unavailable-backend-provenance.tsv")
file(REMOVE "${unavailable_provenance_path}")
execute_process(
    COMMAND "${CLI}" --batch "${batch}" --backend cuda
            --cuda-device 2147483647
            --backend-provenance-out "${unavailable_provenance_path}"
    RESULT_VARIABLE unavailable_result
    OUTPUT_VARIABLE unavailable_stdout
    ERROR_VARIABLE unavailable_stderr)
if(NOT unavailable_result EQUAL 2)
    message(FATAL_ERROR
        "unavailable CUDA provenance returned ${unavailable_result}, expected 2; "
        "stdout=${unavailable_stdout}; stderr=${unavailable_stderr}")
endif()
file(READ "${unavailable_provenance_path}" unavailable_provenance_text)
string(REGEX MATCHALL "[^\n]+" unavailable_provenance_lines
    "${unavailable_provenance_text}")
list(LENGTH unavailable_provenance_lines unavailable_provenance_line_count)
if(NOT unavailable_provenance_line_count EQUAL 2)
    message(FATAL_ERROR
        "unavailable CUDA provenance emitted an unexpected row count: "
        "${unavailable_provenance_text}")
endif()
list(GET unavailable_provenance_lines 1 unavailable_provenance_row)
string(REGEX REPLACE " +" ";" unavailable_provenance_fields
    "${unavailable_provenance_row}")
list(LENGTH unavailable_provenance_fields unavailable_provenance_field_count)
if(NOT unavailable_provenance_field_count EQUAL 7)
    message(FATAL_ERROR
        "unavailable CUDA provenance row is not rectangular: "
        "${unavailable_provenance_row}")
endif()
list(GET unavailable_provenance_fields 0 unavailable_executed)
list(GET unavailable_provenance_fields 1 unavailable_backend)
list(GET unavailable_provenance_fields 2 unavailable_candidate)
list(GET unavailable_provenance_fields 3 unavailable_final)
list(GET unavailable_provenance_fields 4 unavailable_reasons)
list(GET unavailable_provenance_fields 5 unavailable_cpu_adjudicated)
list(GET unavailable_provenance_fields 6 unavailable_source)
if(NOT unavailable_executed STREQUAL "0"
        OR NOT unavailable_backend STREQUAL "cuda"
        OR NOT unavailable_candidate STREQUAL "none"
        OR NOT unavailable_final STREQUAL "none"
        OR NOT unavailable_reasons STREQUAL "1"
        OR NOT unavailable_cpu_adjudicated STREQUAL "0"
        OR NOT unavailable_source STREQUAL "${SOURCE}")
    message(FATAL_ERROR
        "unavailable CUDA provenance lost its fail-closed diagnostics: "
        "${unavailable_provenance_row}")
endif()

execute_process(
    COMMAND "${CLI}" --batch "${batch}" --dsn --sn-random-seed 1
            --backend auto --backend-audit
    RESULT_VARIABLE preexecution_result
    OUTPUT_VARIABLE preexecution_stdout
    ERROR_VARIABLE preexecution_stderr)
if(NOT preexecution_result EQUAL 2)
    message(FATAL_ERROR
        "pre-execution filename rejection returned ${preexecution_result}, expected 2; "
        "stdout=${preexecution_stdout}; stderr=${preexecution_stderr}")
endif()
string(REGEX MATCHALL "[^\n]+" preexecution_lines "${preexecution_stdout}")
list(LENGTH preexecution_lines preexecution_line_count)
if(NOT preexecution_line_count EQUAL 3)
    message(FATAL_ERROR
        "pre-execution rejection emitted an unexpected row count: ${preexecution_stdout}")
endif()
list(GET preexecution_lines 1 preexecution_header)
if(NOT preexecution_header STREQUAL
        "# ok Delta_EW delta_SN dN_vac sn_nF sn_nD Q_SUSY logQ_GUT mZ2 backend_executed selected_backend candidate_tier final_tier adjudication_reasons cpu_adjudicated backend_audit_match slha_path")
    message(FATAL_ERROR
        "pre-execution rejection header changed: ${preexecution_stdout}")
endif()
list(GET preexecution_lines 2 preexecution_row)
string(REGEX REPLACE " +" ";" preexecution_fields "${preexecution_row}")
list(LENGTH preexecution_fields preexecution_field_count)
if(NOT preexecution_field_count EQUAL 17)
    message(FATAL_ERROR
        "pre-execution rejection row is not rectangular: ${preexecution_row}")
endif()
list(GET preexecution_fields 9 preexecution_executed)
list(GET preexecution_fields 10 preexecution_backend)
list(GET preexecution_fields 11 preexecution_candidate)
list(GET preexecution_fields 12 preexecution_final)
list(GET preexecution_fields 15 preexecution_audit_match)
list(GET preexecution_fields 16 preexecution_path)
if(NOT preexecution_executed STREQUAL "0"
        OR NOT preexecution_backend STREQUAL "cpu"
        OR NOT preexecution_candidate STREQUAL "none"
        OR NOT preexecution_final STREQUAL "none"
        OR NOT preexecution_audit_match STREQUAL "-1"
        OR NOT preexecution_path STREQUAL "${SOURCE}")
    message(FATAL_ERROR
        "pre-execution rejection falsely claimed execution: ${preexecution_row}")
endif()

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
