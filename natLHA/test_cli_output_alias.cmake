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
