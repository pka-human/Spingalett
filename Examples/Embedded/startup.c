/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Minimal start-up code for a Cortex-M4F board (QEMU's MPS2 AN386): vector table, data and bss
   initialization, the FPU, and semihosting for printf and the exit status. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);
extern void initialise_monitor_handles(void);
extern void __libc_init_array(void);

void _init(void) {}
void _fini(void) {}

void Reset_Handler(void) {
    if (&_sdata != &_sidata) memcpy(&_sdata, &_sidata, (size_t)((char *)&_edata - (char *)&_sdata));
    memset(&_sbss, 0, (size_t)((char *)&_ebss - (char *)&_sbss));
    *(volatile uint32_t *)0xE000ED88u |= 0xFu << 20;          /* CPACR: full access to the FPU */
    __asm volatile("dsb\n\tisb");
    initialise_monitor_handles();
    __libc_init_array();
    exit(main());
}

/* Any fault ends the program through semihosting with a run-time error status. */
static void Fault_Handler(void) {
    register uint32_t op __asm__("r0") = 0x18;                /* SYS_EXIT */
    register uint32_t reason __asm__("r1") = 0x20023;         /* ADP_Stopped_RunTimeErrorUnknown */
    __asm volatile("bkpt 0xAB" : : "r"(op), "r"(reason));
    for (;;) {}
}

__attribute__((section(".isr_vector"), used))
static void (*const vectors[16])(void) = {
    (void (*)(void))(uintptr_t)&_estack, Reset_Handler,
    Fault_Handler, Fault_Handler, Fault_Handler, Fault_Handler, Fault_Handler,   /* NMI .. UsageFault */
    0, 0, 0, 0, Fault_Handler, Fault_Handler, 0, Fault_Handler, Fault_Handler,    /* SVC, Debug, PendSV, SysTick */
};
