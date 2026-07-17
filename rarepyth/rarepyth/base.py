# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:06:32 2024

@author: Wang Junhao
"""

import os
import sys
import subprocess as sp


try:
    import msvcrt
except ModuleNotFoundError:
    _mswindows = False
else:
    _mswindows = True


class msgstates:

    states_avail = ["Error", "End"]

    def __init__(self, func):
        self.func = func

    def __call__(self, new_states=None):
        if new_states is not None:
            if isinstance(new_states, list):
                self.states_avail = self.states_avail + new_states
            else:
                self.states_avail.append(new_states)
        return self.func(self.states_avail)


class MessageError(Exception):

    def __init__(self, msg):
        self.msg = msg if isinstance(msg, str) else str(msg)

    def __str__(self):
        return self.msg


class Message:

    def __init__(
        self,
        text,
        state=None
    ):
        self.text = text
        self.state = state

    def printm(self):
        if self.state is None:
            print("\nrarepyth Message: {}".format(self.text))
        else:
            print("\nrarepyth Message: [{}]{}".format(self.state, self.text))
        sys.stdout.flush()

    @classmethod
    def readm(cls, filename):
        try:
            new_message = checkcmd("grep rarepyth\\ Message {}"
                                   .format(filename)).split('\n')[-1][18:]
            try:
                state = new_message.split("]")[0].split("[")[1]
                if state in cls.states_avail():
                    return cls(new_message.split("]")[1], state=state)
                else:
                    return cls(new_message)
            except Exception:
                return cls(new_message)
        except sp.CalledProcessError:
            return cls("No message yet")

    def exitm(self):
        self.state = "Error"
        self.printm()
        raise MessageError(self.text)

    @msgstates
    def states_avail(states):
        return states

    @classmethod
    def add_new_states(cls, new_states):
        cls.states_avail(new_states)


if not _mswindows:

    def runcmd(cmd, print_error=True, **kwargs):
        if isinstance(cmd, str):
            shell = True
        elif isinstance(cmd, list):
            shell = False
        else:
            Message("Invalid command format").exitm()

        result = sp.run(
            cmd,
            capture_output=True,
            check=False,
            shell=shell,
            text=True,
            **kwargs
        )

        if print_error and result.stderr:
            Message("Command exited with error: {}"
                    .format(result.stderr)).exitm()

        if result.stdout:
            return result.stdout.strip()
        else:
            return

    def checkcmd(cmd, **kwargs):
        if isinstance(cmd, str):
            shell = True
        elif isinstance(cmd, list):
            shell = False
        else:
            Message("Invalid command format").exitm()

        result = sp.run(
            cmd,
            capture_output=True,
            check=True,
            shell=shell,
            text=True,
            **kwargs
        )

        if result.stdout:
            return result.stdout.strip()
        else:
            return
