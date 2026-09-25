/** Initials from a display name, for avatars. */
export const initials = (fullName: string) =>
	fullName
		.split(" ")
		.filter(Boolean)
		.map((part) => part[0])
		.join("");
