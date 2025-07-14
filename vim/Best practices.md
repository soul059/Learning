---
tags:
  - CodeEditor/vim
---

## Why not use the arrow keys?

Vim is about **efficiency**, which means staying as close as possible to the home row keys.

The arrow keys are intuitive, but using them requires you to move your right hand completely away from the home row keys. This is inefficient.

## Where should I rest my fingers?

You should rest with your fingers on the home row keys, even though this makes your right index finger responsible for hitting both h and j keys.

- Your right hand fingers should rest on `jkl;`
- Your left  hand fingers should rest on `asdf`

## Mapping Caps Lock to Escape

Most Vim users map their Caps Lock key to Escape. This is because it is much easier to reach the Caps Lock key than the Escape key.

If you are using Mac OS, you can map Caps Lock to Escape by going to System Preferences > Keyboard > Modifier Keys and selecting Escape from the dropdown for Caps Lock.

If you are using Windows, you can use software such as [Uncap](https://github.com/susam/uncap) to map Caps Lock to Escape.


## When to use r vs s

The r operator is useful when you want to replace a single character with another character. It's a nice way to quickly correct a typo without leaving normal mode.

The s operator is useful when you want to replace a single character with multiple characters via insert mode.


## When to use f and F motions?

The find motions are useful when you want to quickly jump to a specific character within a line, especially if the character is far from the current position.

These motions are frequently used to jump to symbols such as ( and {.