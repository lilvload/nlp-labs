# -*- coding: utf-8 -*-
"""
English topical texts of four subject areas for studying the dependence
of part-of-speech (POS) tag frequency on the topic of the text.
Each text contains at least 5000 characters.
"""

EN_TEXTS = {

"mathematics": """
Mathematics is a fundamental science that studies quantity, structure, space, and
change through abstract reasoning and rigorous proof. Its main branches include
algebra, geometry, mathematical analysis, number theory, and probability theory.
Each branch operates with its own definitions, axioms, and theorems that follow
logically from one another. Algebra investigates operations on numbers and symbols,
generalizing arithmetic to equations, polynomials, and abstract structures such as
groups, rings, and fields. Geometry studies the properties of figures, distances,
angles, areas, and volumes, as well as transformations that preserve certain
quantities. Mathematical analysis is built on the concepts of limit, continuity,
derivative, and integral, allowing mathematicians to describe continuous change and
to model dynamic processes with remarkable precision.

The derivative of a function characterizes the rate at which that function changes
with respect to its argument. If a function describes the position of a body over
time, then its first derivative equals velocity and its second derivative equals
acceleration. The integral, conversely, allows one to compute an accumulated
quantity, such as the area under a curve or the total effect of continuous change.
The fundamental theorem of calculus connects differentiation and integration,
proving that they are inverse operations. Because of this profound result,
mathematical analysis has become a powerful tool for solving problems in physics,
economics, biology, and engineering.

Probability theory studies the regularities of random events and allows us to
estimate the chances that particular outcomes will occur. The probability of an
event is defined as the ratio of favorable outcomes to the total number of equally
likely outcomes. The expected value of a random variable characterizes its average,
the variance describes the spread of values around the mean, and the standard
deviation measures the typical deviation from the expected value. The law of large
numbers states that as the number of trials increases, the average of the results
approaches the expected value. The central limit theorem explains why the
distribution of a sum of many independent variables tends toward the normal
distribution.

Proof is the central method of mathematical thought. A proof is a sequence of
logical steps, each of which follows from axioms, definitions, or previously proven
statements. We distinguish direct proof, proof by contradiction, and the method of
mathematical induction. Induction allows us to prove a statement for all natural
numbers by establishing a base case and an inductive step. The rigor of proofs
distinguishes mathematics from other sciences, since mathematical truth does not
depend on experiment but follows solely from logic and careful definition.

Modern mathematics is widely applied in cryptography, machine learning, signal
processing, and optimization. Numerical methods allow us to approximate solutions to
equations that have no closed analytic form. Linear algebra, which operates with
vectors and matrices, lies at the heart of computer graphics and artificial
intelligence algorithms. Set theory provides the foundation upon which the whole of
modern mathematics is built, while mathematical logic studies the very nature of
reasoning and proof. Thus mathematics remains the universal language of science,
connecting abstract theoretical constructions with practical applications across an
extraordinary range of human activities. The development of computing has only
strengthened the role of mathematics, because complex models demand precise
formulas, efficient algorithms, and reliable methods for verifying the results.

Geometry developed from the visual constructions of the ancient Greeks to the
abstract spaces of modern topology. Euclidean geometry describes flat space, in
which exactly one parallel line passes through a point not on a given line. The
discovery of non-Euclidean geometries showed that other, equally consistent systems
are possible, in which this property is violated. Analytic geometry, introduced by
Descartes, combined algebra with geometry, allowing figures to be described by
equations in a coordinate system. Thanks to this, the line, the circle, the
parabola, and other curves received a precise algebraic description, which greatly
simplified the solution of geometric problems and opened the way to calculus.

Solving equations is one of the oldest and most important tasks of mathematics.
Linear equations describe simple relationships, quadratic equations have a known
formula for their roots through the discriminant, and equations of higher degree
require more sophisticated methods. Systems of equations allow us to describe
interrelated conditions and to find the values of several unknowns at once.
Differential equations relate a function to its derivatives and describe the laws of
motion, oscillation, heat transfer, and many other processes. The ability to build
and solve equations is a key skill for every engineer, physicist, and economist,
because it is the equation that turns the verbal description of a problem into a
precise mathematical model suitable for computation and rigorous analysis.
""",

"physics": """
Physics is a natural science that studies the most general properties of matter,
energy, and the fundamental interactions between bodies. It seeks to describe the
laws of nature through mathematical relationships that can be verified by
experiment. Classical mechanics, founded by Isaac Newton, describes the motion of
bodies under the action of forces. The first law states that a body remains at rest
or in uniform straight-line motion until external forces act upon it. The second law
relates the acceleration of a body to the applied force and its mass. The third law
states that every action is met by an equal and opposite reaction. These laws allow
us to calculate the orbits of planets, the motion of vehicles, and the flight of
projectiles with great accuracy.

Energy is one of the most important physical quantities. We distinguish kinetic
energy, which depends on the speed of a body, and potential energy, which depends on
its position within a field of forces. The law of conservation of energy states that
the total energy of a closed system remains constant; energy merely transforms from
one form into another. The work done by a force equals the change in the energy of
the body, while power characterizes the rate at which work is performed. These
concepts form the basis for analyzing mechanical, thermal, and electrical processes.

Thermodynamics studies thermal phenomena and the transformation of energy. The first
law of thermodynamics is a form of the conservation of energy applied to thermal
processes. The second law introduces the concept of entropy and states that the
entropy of an isolated system cannot decrease. This explains the irreversibility of
many natural processes, in particular the transfer of heat from a hot body to a cold
one. The kinetic theory of gases relates macroscopic parameters such as pressure and
temperature to the motion of individual molecules.

Electromagnetism describes the interaction of charged particles and electromagnetic
fields. Coulomb's law defines the force between stationary charges. The motion of
charges constitutes an electric current, which in turn produces a magnetic field.
Maxwell's equations unite electric and magnetic phenomena into a single theory and
predict the existence of electromagnetic waves that propagate at the speed of light.
Visible light, radio waves, and X-rays are all manifestations of electromagnetic
waves of different wavelengths.

Quantum mechanics describes the behavior of microscopic objects such as electrons
and photons. It states that energy can be transferred only in discrete portions
called quanta. Heisenberg's uncertainty principle states that it is impossible to
measure both the position and the momentum of a particle exactly at the same time.
The wave function describes the state of a quantum system and allows us to compute
the probabilities of measurement outcomes. Einstein's theory of relativity changed
our understanding of space and time, showing that they form a single space-time
continuum and that mass and energy are equivalent. Optics, the study of light,
explains reflection, refraction, interference, and diffraction. Modern physics
continues to investigate elementary particles, dark matter, and the origin of the
universe, combining bold theoretical models with precise experiments carried out on
particle accelerators and space telescopes.

Oscillations and waves are an important topic of physics, since they describe a wide
range of phenomena, from the motion of a pendulum to the propagation of sound and
light. Harmonic oscillations are characterized by amplitude, period, frequency, and
phase. Resonance occurs when the frequency of an external force coincides with the
natural frequency of an oscillating system, leading to a sharp increase in
amplitude. A wave transfers energy without transferring matter, and we distinguish
longitudinal and transverse waves. Sound waves are longitudinal oscillations of the
particles of a medium, whereas electromagnetic waves are transverse and can
propagate even in a vacuum.

Atomic and nuclear physics investigate the structure of matter at the deepest level.
An atom consists of a positively charged nucleus and electrons that move around it.
The nucleus is formed of protons and neutrons, which are held together by powerful
nuclear forces. Radioactive decay is accompanied by the emission of particles and
energy, while nuclear reactions of fission and fusion are able to release enormous
amounts of energy. It is precisely these processes that underlie the operation of
nuclear power plants and the prospects of controlled thermonuclear fusion, which
could provide humanity with a practically inexhaustible source of clean energy.
Experiment remains the foundation of physical knowledge, for it is the experiment
that tests theoretical predictions and reveals new phenomena yet unknown to science. Large research facilities such as particle
accelerators and gravitational observatories open new horizons before researchers,
and the careful measurement of tiny signals demands extraordinary precision and
patience from the scientists who pursue the secrets of nature.
""",

"art": """
Art is a special form of human creativity that reflects the world through artistic
images and evokes aesthetic experiences. It encompasses painting, sculpture,
architecture, music, theatre, literature, and cinema. Each form of art employs its
own expressive means, yet all of them strive to convey beauty, emotion, and meaning.
Painting uses color, line, light, and composition to create a visual image. Artists
blend shades of paint on the canvas, conveying the mood, character, and inner world
of the figures they depict. The great painters have left humanity a priceless legacy
that continues to inspire new generations of creators around the world.

The history of art spans many eras and styles. Ancient art celebrated the harmony
and proportion of the human body. The Gothic style directed the gaze upward toward
the heavens through soaring vaults and delicate stained glass. The Renaissance
renewed interest in the human being, dignity, and beauty, uniting scientific inquiry
with artistic mastery. The Baroque astonished its contemporaries with splendor,
dynamism, and the dramatic play of light and shadow. Later came classicism,
romanticism, realism, impressionism, and the many movements of modern art, each
reinterpreting the role of the artist and the purpose of creativity in fresh ways.

Music is the art of sounds organized in time. Melody, harmony, rhythm, and timbre
form the musical fabric, which is capable of conveying the subtlest shades of
feeling. Composers create symphonies, operas, sonatas, and songs that sound in
concert halls and in the homes of listeners. Music accompanies people throughout
their lives, uniting them during celebrations, soothing them in moments of sorrow,
and inspiring them toward great achievements. The skill of performers transforms
written notes into living sound, filled with personal experience and deep emotion.

Theatre combines literature, acting, stage design, and music into a single
performance. Actors embody the images of characters, conveying to the audience the
thoughts and feelings of the playwright. The stage becomes a space where conflicts,
passions, and human destinies come alive. The spectator, sympathizing with the
heroes, reflects on his own life and moral values. Theatrical art has an ancient
history, stretching from the tragedies of antiquity to modern experimental
performances that challenge our expectations.

Art performs important social functions. It preserves the cultural memory of a
people, shapes aesthetic taste, cultivates a sense of the beautiful, and promotes
understanding among people of different cultures. Works of art prompt us to reflect
on the eternal questions of existence, love, goodness, and justice. Architecture
unites art and engineering, creating spaces for human life, while literature opens
boundless worlds of imagination through the art of the written word. Contemporary
art actively uses new technologies, combining traditional means with digital media,
video installations, and interactive projects. Despite changing forms, the essence
of art remains unchanged: it seeks to reveal to the human being the beauty of the
world and the depth of the soul, granting inspiration and the joy of creative
understanding of reality.

Sculpture embodies an artistic idea in three-dimensional form, giving stone, bronze,
or wood the features of living expressiveness. The master frees the image from the
shapeless block, drawing nearer to perfection with every movement of the chisel.
Free-standing sculpture allows the work to be viewed from all sides, while relief
combines volume with planar depiction. Monuments immortalize notable events and
outstanding individuals, adorning the squares and parks of cities. Decorative and
applied art brings beauty into everyday life through embroidery, ceramics, carving,
and weaving, preserving centuries-old folk traditions and granting people a sense of
harmony and comfort in their daily surroundings.

Cinema became one of the youngest and most influential forms of art of the twentieth
century. It combines image, sound, acting, and editing, creating a dynamic narrative
that captivates millions of viewers. The director guides the entire creative
process, realizing a personal vision on the screen. Cinematography, the musical
score, and special effects intensify the emotional impact of the film. Cinema is
able to transport the viewer into the past or the future, to distant lands and
fantastic worlds, prompting reflection on the important problems of our time. The
union of art and technology makes cinema a unique cultural phenomenon that is
constantly developing and reinventing the ways in which stories can be told. Throughout the centuries, artists have responded to the spirit of
their age, reflecting its hopes, fears, and aspirations, so that each work of art
becomes a window into the soul of the people who created it and a bridge that
connects distant generations across the long passage of time. The museum, the gallery, and the concert hall remain places where
people of every background can pause from the noise of daily routine and encounter
something larger than themselves, a shared inheritance of human imagination that
belongs equally to all.
""",

"fiction": """
The sun was slowly setting beyond the horizon, flooding the sky with a warm, golden
glow. An old fisherman was returning home along a narrow path that wound between
tall grasses. He carried heavy nets on his shoulder, and a quiet joy shone in his
weary eyes. The sea had been generous today, and his boat had come back with a rich
catch. The wind gently ruffled the old man's grey hair, whispering ancient stories
of distant voyages and unexplored shores. Somewhere far away, gulls were crying,
bidding farewell to the fading day as the first stars appeared.

A small village nestled in a valley between gentle hills. Here lived simple,
hardworking people who loved their land and honored their old customs. In the
morning, when the first rays of the sun touched the tops of the trees, the women
lit their stoves, and fragrant smoke rose above the cottages. Children ran to the
stream, laughing and splashing water at one another. The old folk sat on benches by
the fences and told their grandchildren tales of brave heroes and magical birds that
bring happiness to those with a pure and gentle heart.

A young woman stood by the window and watched the first snowflakes slowly whirling
in the cold air. Her heart filled with an unaccountable anxiety and, at the same
time, a quiet hope. She thought of the one who had gone far away and promised to
return in the spring. The winter seemed endless; each day dragged on like a year.
And yet she believed that love was stronger than distance and time, that the warm
days of spring would surely come, and that familiar footsteps would once again echo
across the snow-covered yard she knew so well.

The forest greeted the traveler with coolness and a mysterious silence. Tall pines
rose toward the very sky, and their roots intertwined like the hands of old friends.
Soft light filtered between the trees, falling in patches upon the moss-covered
ground. The traveler stopped, listening to the whisper of leaves and the song of
unseen birds. It seemed to him that the forest was alive, that it watched his every
step attentively and kept an ancient wisdom inaccessible to the restless human mind.
He felt small and humble beneath those towering trees.

Years passed, the seasons changed, and yet life in the small town flowed on in its
usual course. On market days the streets filled with chatter, laughter, and the
smell of fresh bread. Lovers strolled along the embankment, holding hands and
dreaming of a shared future. Mothers rocked their infants, singing old lullabies.
And above all of this, unchanging and eternal, stretched the high sky that had
witnessed countless human destinies, joys, and sorrows. It seemed that in this
simple flow of everyday life lay the deepest secret of human happiness, which needs
no loud words but lives in the small, warm moments that come together into one
great and unrepeatable story of every soul. One spring, a wandering musician came
to the village and played a melody so enchanting that the people, one by one, left
their work and gathered around him to listen in wonder and gratitude.

At night a vast starry sky stretched over the village. Thousands of shimmering
lights were reflected in the calm water of the pond, and it seemed that earth and
heaven had merged into a single whole. Somewhere in the grass the crickets chirped,
and a gentle breeze carried the scent of freshly mown hay. A boy lay on his back and
gazed upward, trying to count the stars. He dreamed of distant worlds, of brave
voyages and the unexplored mysteries of the universe. His heart pounded with the
anticipation of a great and beautiful future that surely awaited him somewhere
ahead, beyond the horizon of ordinary days.

Then autumn came, and the trees dressed themselves in gold and crimson. The leaves
slowly fell, covering the ground with a colorful carpet. The air smelled of apples,
mushrooms, and the smoke of distant bonfires. The villagers gathered the harvest,
filling their barns with the fruits of their labor. Cranes called out their
farewells in the sky as they flew away to warmer lands. In this gentle sadness of
the autumn parting there was something solemn and wise, as if nature itself reminded
people of the eternal cycle of life, of the inevitability of change, and of the
quiet beauty of every passing moment that is worth keeping in the heart and passing
on to those who will come after us, as the most precious treasure of all. And when winter finally arrived, the village was covered with soft
white snow that sparkled beneath the moonlight like scattered diamonds, while warm
lights glowed in the windows and the families gathered around the stove, listening
to the howling of the storm and to the gentle voice of the grandmother telling her
old and beloved tales. Spring returned once more, the orchards bloomed in white, and the
larks sang high above the golden fields, while the people, grateful for every fair
day, found in the simple round of the seasons a beauty quiet and enduring, like life
itself, which always triumphs and is always reborn anew.
""",
}

if __name__ == "__main__":
    for theme, text in EN_TEXTS.items():
        print(theme, len(text), "chars")
