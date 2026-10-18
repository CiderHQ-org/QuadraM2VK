            // Scheduled once the CPU finished scheduling it (`kernelEndTime`): submitted, or held
            // for its pass's kernel submission (`CompletionPolicy::Grouped`), which the sleep
            // below outlasts. Its kernel start is stored as scheduling begins, before its batch is
            // replayed and submitted.
            while command_buffer.kernel_end_time() == 0.0 {
