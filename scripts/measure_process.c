/* GPL-3.0-only. Linux child wall/CPU time and peak RSS, without an external time package. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double seconds(struct timespec t) { return t.tv_sec + t.tv_nsec / 1e9; }
static double cpu_seconds(struct timeval t) { return t.tv_sec + t.tv_usec / 1e6; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s output.json command [args...]\n", argv[0]); return 2; }
    struct timespec start, end;
    if (clock_gettime(CLOCK_MONOTONIC, &start)) { perror("clock_gettime"); return 1; }
    pid_t child = fork();
    if (child < 0) { perror("fork"); return 1; }
    if (!child) { execvp(argv[2], argv + 2); perror("execvp"); _exit(127); }
    struct rusage usage;
    int status;
    while (wait4(child, &status, 0, &usage) < 0) {
        if (errno != EINTR) { perror("wait4"); return 1; }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end)) { perror("clock_gettime"); return 1; }
    FILE *output = fopen(argv[1], "w");
    if (!output) { perror("fopen"); return 1; }
    int written = fprintf(output, "{\"wall_seconds\":%.9f,\"user_seconds\":%.6f,\"system_seconds\":%.6f,\"peak_rss_kib\":%ld}\n",
        seconds(end) - seconds(start), cpu_seconds(usage.ru_utime), cpu_seconds(usage.ru_stime), usage.ru_maxrss);
    if (fclose(output) || written < 0) { perror("writing measurements"); return 1; }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
}
