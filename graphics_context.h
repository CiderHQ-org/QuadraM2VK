#ifndef MACOBLOX_GRAPHICS_CONTEXT_H
#define MACOBLOX_GRAPHICS_CONTEXT_H

/* EGL outcomes are recorded before Darling discards them in its CGL layer. */
typedef struct {
    void *previous;
    unsigned long binding_serial;
    void **surface_slot;
    void *previous_surface;
} macoblox_context_change;

int macoblox_begin_context_change(macoblox_context_change *change);
int macoblox_finish_context_change(const macoblox_context_change *change, int cgl_error);
void macoblox_record_egl_binding(unsigned int succeeded, void *context);
void macoblox_register_egl_context(void *context);
void macoblox_forget_egl_context(void *context);
int macoblox_graphics_first_draw(void);
void macoblox_record_egl_swap(unsigned int succeeded);
unsigned long macoblox_egl_swap_serial(void);
int macoblox_finish_cgl_swap(unsigned long serial, int cgl_error);
int macoblox_bind_desktop_gl(void);
int macoblox_egl_binding_error(void);
int macoblox_capture_egl_error(void);
int macoblox_graphics_get_error(void);

/* Return an owned, terminated copy, or NULL on allocation/invalid input. */
int *macoblox_copy_pixel_attributes(const int *attributes);

#endif
