/* Linux libc compatibility for runtimes requiring their own /proc/self/maps.
 * The trusted parent sends only the calling process's read-only maps descriptor.
 * No path or PID is accepted from this client; all other opens use libc normally.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int own_maps(const char *path) {
    if (!path || strcmp(path, "/proc/self/maps")) return -2;
    const char *endpoint = getenv("ATREX_PROC_MAPS_SOCKET");
    if (!endpoint) return -2;
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (strlen(endpoint) >= sizeof(address.sun_path)) { errno = ENAMETOOLONG; return -1; }
    strcpy(address.sun_path, endpoint);
    int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock < 0) return -1;
    if (connect(sock, (struct sockaddr *)&address, sizeof(address))) { close(sock); return -1; }
    if (send(sock, "M\n", 2, 0) != 2) { close(sock); return -1; }
    char byte, control[CMSG_SPACE(sizeof(int))];
    struct iovec data = {.iov_base = &byte, .iov_len = 1};
    struct msghdr message = {.msg_iov = &data, .msg_iovlen = 1,
                            .msg_control = control, .msg_controllen = sizeof(control)};
    int result = -1;
    if (recvmsg(sock, &message, MSG_CMSG_CLOEXEC) == 1) {
        struct cmsghdr *c = CMSG_FIRSTHDR(&message);
        if (c && c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS &&
            c->cmsg_len == CMSG_LEN(sizeof(int))) memcpy(&result, CMSG_DATA(c), sizeof(int));
    }
    close(sock);
    if (result < 0) errno = EACCES;
    return result;
}

#define DEFINE_OPEN(NAME) \
int NAME(const char *path, int flags, ...) { \
    int fd = own_maps(path); if (fd != -2) return fd; \
    mode_t mode = 0; \
    if ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE)) { \
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); \
    } \
    int (*original)(const char *, int, ...) = dlsym(RTLD_NEXT, #NAME); \
    return original(path, flags, mode); \
}
#define DEFINE_OPENAT(NAME) \
int NAME(int dirfd, const char *path, int flags, ...) { \
    int fd = own_maps(path); if (fd != -2) return fd; \
    mode_t mode = 0; \
    if ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE)) { \
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); \
    } \
    int (*original)(int, const char *, int, ...) = dlsym(RTLD_NEXT, #NAME); \
    return original(dirfd, path, flags, mode); \
}
DEFINE_OPEN(open)
DEFINE_OPEN(open64)
DEFINE_OPENAT(openat)
DEFINE_OPENAT(openat64)

#define DEFINE_FOPEN(NAME) \
FILE *NAME(const char *path, const char *mode) { \
    int fd = own_maps(path); if (fd != -2) return fd < 0 ? NULL : fdopen(fd, "r"); \
    FILE *(*original)(const char *, const char *) = dlsym(RTLD_NEXT, #NAME); \
    return original(path, mode); \
}
DEFINE_FOPEN(fopen)
DEFINE_FOPEN(fopen64)

/* glibc's main-thread pthread_getattr_np uses an internal fopen which cannot
 * be interposed. Recover the actual stack VMA from the same caller-only broker;
 * keep libc's result for every other thread and for successful native procfs.
 */
#include <pthread.h>
#include <stdint.h>
#include <sys/resource.h>
#include <sys/syscall.h>
int pthread_getattr_np(pthread_t thread, pthread_attr_t *attributes) {
    int (*original)(pthread_t, pthread_attr_t *) = dlsym(RTLD_NEXT, "pthread_getattr_np");
    int error = original(thread, attributes);
    if (!error || syscall(SYS_gettid) != getpid() || !pthread_equal(thread, pthread_self()))
        return error;
    int fd = own_maps("/proc/self/maps");
    if (fd < 0) return error;
    FILE *maps = fdopen(fd, "r");
    if (!maps) { close(fd); return error; }
    uintptr_t pointer = (uintptr_t)&maps, previous_end = 0;
    unsigned long low, high;
    char line[4096];
    struct rlimit limit;
    while (fgets(line, sizeof(line), maps)) {
        if (sscanf(line, "%lx-%lx", &low, &high) != 2) continue;
        if (low <= pointer && pointer < high && !getrlimit(RLIMIT_STACK, &limit)) {
            size_t size = high - previous_end;
            if (limit.rlim_cur != RLIM_INFINITY && limit.rlim_cur < size) size = limit.rlim_cur;
            fclose(maps);
            pthread_attr_init(attributes);
            pthread_attr_setguardsize(attributes, 0);
            return pthread_attr_setstack(attributes, (void *)(high - size), size);
        }
        previous_end = high;
    }
    fclose(maps);
    return error;
}

/* Bun resolves open directory handles using /proc/self/fd/N. Return only a
 * textual link for the caller's own handle, never an openable host /proc link. */
static ssize_t own_link(const char *path, char *buffer, size_t size) {
    const char *prefix = "/proc/self/fd/";
    if (strncmp(path, prefix, strlen(prefix))) return -2;
    const char *number = path + strlen(prefix);
    if (!*number || strlen(number) > 10 || strspn(number, "0123456789") != strlen(number)) {
        errno = EINVAL; return -1;
    }
    const char *endpoint = getenv("ATREX_PROC_MAPS_SOCKET");
    if (!endpoint) return -2;
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (strlen(endpoint) >= sizeof(address.sun_path)) { errno = ENAMETOOLONG; return -1; }
    strcpy(address.sun_path, endpoint);
    int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock < 0) return -1;
    if (connect(sock, (struct sockaddr *)&address, sizeof(address))) { close(sock); return -1; }
    char request[16]; int count = snprintf(request, sizeof(request), "R%s\n", number);
    if (send(sock, request, count, 0) != count) { close(sock); return -1; }
    size_t got = 0;
    while (got < size) {
        ssize_t part = recv(sock, buffer + got, size - got, 0);
        if (part <= 0) break;
        got += part;
    }
    close(sock);
    if (!got) { errno = ENOENT; return -1; }
    return (ssize_t)got;
}
ssize_t readlink(const char *path, char *buffer, size_t size) {
    ssize_t result = own_link(path, buffer, size); if (result != -2) return result;
    ssize_t (*original)(const char *, char *, size_t) = dlsym(RTLD_NEXT, "readlink");
    return original(path, buffer, size);
}
ssize_t readlinkat(int directory, const char *path, char *buffer, size_t size) {
    ssize_t result = own_link(path, buffer, size); if (result != -2) return result;
    ssize_t (*original)(int, const char *, char *, size_t) = dlsym(RTLD_NEXT, "readlinkat");
    return original(directory, path, buffer, size);
}
