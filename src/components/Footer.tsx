// src/components/Footer.tsx
export const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t mt-12 py-6">
      <div className="container mx-auto px-4 text-center text-gray-500">
        <p>&copy; {currentYear} My Velite Blog. All Rights Reserved.</p>
      </div>
    </footer>
  );
};
