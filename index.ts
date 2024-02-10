const world = "world";

export function hello(who: string = world): void {
  console.log(`Hello, ${who}!`);
}

hello();
