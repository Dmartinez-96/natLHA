cmake_minimum_required(VERSION 3.10)

foreach(required CLI SOURCE WORK)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "missing required test argument: ${required}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${WORK}")
set(spectrum "${WORK}/spectrum.slha")
set(batch "${WORK}/batch.txt")
configure_file("${SOURCE}" "${spectrum}" COPYONLY)
file(SHA256 "${spectrum}" before_hash)
file(WRITE "${batch}" "${spectrum}\n")

execute_process(
    COMMAND "${CLI}" --batch "${batch}" --out "${spectrum}"
    RESULT_VARIABLE cli_result
    OUTPUT_VARIABLE cli_stdout
    ERROR_VARIABLE cli_stderr
)

file(SHA256 "${spectrum}" after_hash)
if(NOT before_hash STREQUAL after_hash)
    message(FATAL_ERROR "batch output alias modified the listed spectrum")
endif()
if(NOT cli_result EQUAL 1)
    message(FATAL_ERROR
        "batch output alias returned ${cli_result}, expected usage failure 1; "
        "stdout=${cli_stdout}; stderr=${cli_stderr}")
endif()
string(FIND "${cli_stderr}" "resolves to a spectrum in the batch list" diagnostic_position)
if(diagnostic_position EQUAL -1)
    message(FATAL_ERROR "batch output alias lost its distinct diagnostic: ${cli_stderr}")
endif()

execute_process(
    COMMAND "${CLI}" --batch "${batch}" --backend auto
            --backend-provenance-out "${spectrum}"
    RESULT_VARIABLE provenance_spectrum_result
    OUTPUT_VARIABLE provenance_spectrum_stdout
    ERROR_VARIABLE provenance_spectrum_stderr
)
file(SHA256 "${spectrum}" provenance_spectrum_hash)
if(NOT before_hash STREQUAL provenance_spectrum_hash)
    message(FATAL_ERROR "backend provenance alias modified the listed spectrum")
endif()
if(NOT provenance_spectrum_result EQUAL 1
        OR NOT provenance_spectrum_stderr MATCHES
            "backend-provenance-out FILE resolves to a spectrum")
    message(FATAL_ERROR
        "backend provenance spectrum alias was not rejected distinctly: "
        "result=${provenance_spectrum_result}; stderr=${provenance_spectrum_stderr}")
endif()

file(SHA256 "${batch}" batch_hash)
execute_process(
    COMMAND "${CLI}" --batch "${batch}" --backend auto
            --backend-provenance-out "${batch}"
    RESULT_VARIABLE provenance_list_result
    OUTPUT_VARIABLE provenance_list_stdout
    ERROR_VARIABLE provenance_list_stderr
)
file(SHA256 "${batch}" batch_after_hash)
if(NOT batch_hash STREQUAL batch_after_hash)
    message(FATAL_ERROR "backend provenance alias modified the batch list")
endif()
if(NOT provenance_list_result EQUAL 1
        OR NOT provenance_list_stderr MATCHES
            "backend-provenance-out FILE resolves to the input file")
    message(FATAL_ERROR
        "backend provenance list alias was not rejected distinctly: "
        "result=${provenance_list_result}; stderr=${provenance_list_stderr}")
endif()

set(shared_output "${WORK}/shared-output.tsv")
file(REMOVE "${shared_output}")
execute_process(
    COMMAND "${CLI}" --batch "${batch}" --backend auto
            --out "${shared_output}"
            --backend-provenance-out "${shared_output}"
    RESULT_VARIABLE shared_output_result
    OUTPUT_VARIABLE shared_output_stdout
    ERROR_VARIABLE shared_output_stderr
)
if(EXISTS "${shared_output}")
    message(FATAL_ERROR "aliased primary/provenance output was created")
endif()
if(NOT shared_output_result EQUAL 1
        OR NOT shared_output_stderr MATCHES "resolve to the same path")
    message(FATAL_ERROR
        "primary/provenance output alias was not rejected distinctly: "
        "result=${shared_output_result}; stderr=${shared_output_stderr}")
endif()

set(preserved_output "${WORK}/preserved-primary.tsv")
file(WRITE "${preserved_output}" "preserve-me\n")
file(SHA256 "${preserved_output}" preserved_output_hash)
execute_process(
    COMMAND "${CLI}" --batch "${batch}" --backend auto
            --out "${preserved_output}"
            --backend-provenance-out "${WORK}"
    RESULT_VARIABLE provenance_open_failure_result
    OUTPUT_VARIABLE provenance_open_failure_stdout
    ERROR_VARIABLE provenance_open_failure_stderr
)
file(SHA256 "${preserved_output}" after_open_failure_hash)
if(NOT preserved_output_hash STREQUAL after_open_failure_hash)
    message(FATAL_ERROR
        "provenance open failure modified the deferred primary output")
endif()
if(NOT provenance_open_failure_result EQUAL 1
        OR NOT provenance_open_failure_stderr MATCHES
            "cannot write backend provenance file")
    message(FATAL_ERROR
        "provenance open failure did not fail distinctly: "
        "result=${provenance_open_failure_result}; "
        "stderr=${provenance_open_failure_stderr}")
endif()

if(EXISTS "/dev/full")
    execute_process(
        COMMAND "${CLI}" --batch "${batch}" --backend auto
                --out "${preserved_output}"
                --backend-provenance-out "/dev/full"
        RESULT_VARIABLE provenance_write_failure_result
        OUTPUT_VARIABLE provenance_write_failure_stdout
        ERROR_VARIABLE provenance_write_failure_stderr
    )
    file(SHA256 "${preserved_output}" after_write_failure_hash)
    if(NOT preserved_output_hash STREQUAL after_write_failure_hash)
        message(FATAL_ERROR
            "provenance write failure modified the deferred primary output")
    endif()
    if(NOT provenance_write_failure_result EQUAL 1
            OR NOT provenance_write_failure_stderr MATCHES
                "cannot write backend provenance file")
        message(FATAL_ERROR
            "provenance write failure did not fail distinctly: "
            "result=${provenance_write_failure_result}; "
            "stderr=${provenance_write_failure_stderr}")
    endif()

    set(primary_failure_provenance "${WORK}/primary-write-failure-provenance.tsv")
    execute_process(
        COMMAND "${CLI}" --batch "${batch}" --backend auto
                --out "/dev/full"
                --backend-provenance-out "${primary_failure_provenance}"
        RESULT_VARIABLE primary_write_failure_result
        OUTPUT_VARIABLE primary_write_failure_stdout
        ERROR_VARIABLE primary_write_failure_stderr
    )
    if(NOT primary_write_failure_result EQUAL 1
            OR NOT primary_write_failure_stderr MATCHES
                "failed while writing output file: /dev/full")
        message(FATAL_ERROR
            "primary output write failure did not fail distinctly: "
            "result=${primary_write_failure_result}; "
            "stderr=${primary_write_failure_stderr}")
    endif()
endif()
